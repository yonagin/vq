import os
import sys
sys.path.append(os.getcwd())
import glob
from metrics.UTMOS import UTMOSScore
from metrics.periodicity import calculate_periodicity_metrics
import torchaudio
from pesq import pesq
import numpy as np
import torch
import math
from pystoi import stoi
from pathlib import Path
from tqdm import tqdm
from contextlib import contextmanager

import importlib
from omegaconf import OmegaConf
import argparse

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@contextmanager
def suppress_tqdm():
    """抑制第三方库内部的 tqdm 进度条"""
    import tqdm.std as tqdm_std
    original_init = tqdm_std.tqdm.__init__
    
    def patched_init(self, *args, **kwargs):
        kwargs['disable'] = True
        original_init(self, *args, **kwargs)
    
    tqdm_std.tqdm.__init__ = patched_init
    try:
        yield
    finally:
        tqdm_std.tqdm.__init__ = original_init


def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        import yaml
        print(yaml.dump(OmegaConf.to_container(config)))
    return config


def load_vqgan_new(config, ckpt_path=None, is_gumbel=False):
    model = instantiate_from_config(config.model)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "class_path" in config:
        raise KeyError("Expected key `class_path` to instantiate.")
    return get_obj_from_str(config["class_path"])(**config.get("init_args", dict()))


def main(args):
    config_data = OmegaConf.load(args.config_file)
    config_data.data.init_args.batch_size = args.batch_size
    
    dataset = instantiate_from_config(config_data.data)
    dataset.prepare_data()
    dataset.setup()
    
    config_model = load_config(args.config_file, display=False)
    model = load_vqgan_new(config_model, ckpt_path=args.ckpt_path).to(DEVICE)
    codebook_size = model.quantize.n_e
    
    usage = {i: 0 for i in range(codebook_size)}
    
    # 预先初始化 UTMOS
    UTMOS = UTMOSScore(device=DEVICE)
    
    # 初始化指标累加器
    utmos_sumgt = 0
    utmos_sumencodec = 0
    pesq_sumpre = 0
    f1score_sumpre = 0
    stoi_sumpre = []
    f1score_filt = 0
    
    # 获取数据加载器
    test_dataloader = list(dataset._test_dataloader())
    
    # 使用单一进度条
    pbar = tqdm(test_dataloader, desc="Evaluating", unit="sample", ncols=120)
    
    with torch.no_grad():
        for i, batch in enumerate(pbar):
            assert batch["waveform"].shape[0] == 1
            audio_path_str = batch["audio_path"][0]
            
            audio = batch["waveform"].to(DEVICE)
        
            # ========== 编码解码 ==========
            if model.use_ema:
                with model.ema_scope():
                    quant, diff, indices, _ = model.encode(audio)
                    reconstructed_audios = model.decode(quant)
            else:
                quant, diff, indices, _ = model.encode(audio)
                reconstructed_audios = model.decode(quant)
               
            for index in indices.flatten():
                usage[index.item()] += 1
                
            # 保存重建音频
            save_path = args.ckpt_path.parent / "recons" / audio_path_str
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(
                save_path.as_posix(), 
                reconstructed_audios[0].cpu().clip(min=-0.99, max=0.99), 
                sample_rate=24000, encoding='PCM_S', bits_per_sample=16
            )
            
            # ========== 计算指标 ==========
            rawwav, rawwav_sr = torchaudio.load(os.path.join(os.environ['DATA_ROOT'], audio_path_str))
            prewav, prewav_sr = torchaudio.load(save_path.as_posix())
            
            rawwav = rawwav.to(DEVICE)
            prewav = prewav.to(DEVICE)
       
            rawwav_16k = torchaudio.functional.resample(rawwav, orig_freq=rawwav_sr, new_freq=16000)
            prewav_16k = torchaudio.functional.resample(prewav, orig_freq=prewav_sr, new_freq=16000)

            # 1. UTMOS (抑制内部进度条)
            with suppress_tqdm():
                raw_score = UTMOS.score(rawwav_16k)[0].item()
                pre_score = UTMOS.score(prewav_16k)[0].item()

            utmos_sumgt += raw_score
            utmos_sumencodec += pre_score

            # 2. PESQ  
            min_len = min(rawwav_16k.size()[1], prewav_16k.size()[1])
            rawwav_16k_pesq = rawwav_16k[:, :min_len].squeeze(0)
            prewav_16k_pesq = prewav_16k[:, :min_len].squeeze(0)
            pesq_score = pesq(16000, rawwav_16k_pesq.cpu().numpy(), prewav_16k_pesq.cpu().numpy(), "wb", on_error=1)
            pesq_sumpre += pesq_score

            # 3. F1-score
            rawwav_16k_f1 = rawwav_16k[:, :min_len]
            prewav_16k_f1 = prewav_16k[:, :min_len]
            periodicity_loss, pitch_loss, f1_score = calculate_periodicity_metrics(rawwav_16k_f1, prewav_16k_f1)
            
            if math.isnan(f1_score):
                f1score_filt += 1
            else:
                f1score_sumpre += f1_score

            # 4. STOI
            min_len_stoi = min(rawwav.size()[1], prewav.size()[1])
            rawwav_stoi = rawwav[:, :min_len_stoi].squeeze(0)
            prewav_stoi = prewav[:, :min_len_stoi].squeeze(0)
            tmp_stoi = stoi(rawwav_stoi.cpu(), prewav_stoi.cpu(), rawwav_sr, extended=False)
            stoi_sumpre.append(tmp_stoi)
            
            # 更新进度条显示当前平均指标
            n = i + 1
            pbar.set_postfix({
                'UTMOS': f'{utmos_sumencodec/n:.2f}',
                'PESQ': f'{pesq_sumpre/n:.2f}',
                'F1': f'{f1score_sumpre/max(1, n-f1score_filt):.2f}',
                'STOI': f'{np.mean(stoi_sumpre):.3f}'
            })
    
    # ========== 最终结果 ==========
    num_samples = len(stoi_sumpre)
    num_count = sum(1 for v in usage.values() if v > 0)
    utilization = num_count / codebook_size
    
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    
    def print_and_save(message, file):
        print(message)  
        file.write(message + '\n') 
        
    with open(Path(args.ckpt_path).parent / "result.txt", 'w') as f:
        print_and_save(f"UTMOS_raw:     {utmos_sumgt/num_samples:.4f}", f)
        print_and_save(f"UTMOS_encodec: {utmos_sumencodec/num_samples:.4f}", f)
        print_and_save(f"PESQ:          {pesq_sumpre/num_samples:.4f}", f)
        print_and_save(f"F1_score:      {f1score_sumpre/(num_samples-f1score_filt):.4f} (filtered: {f1score_filt})", f)
        print_and_save(f"STOI:          {np.mean(stoi_sumpre):.4f}", f)
        print_and_save(f"Utilization:   {utilization:.4f} ({num_count}/{codebook_size})", f)
    

def get_args():
    parser = argparse.ArgumentParser(description="inference parameters")
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--ckpt_path", required=True, type=Path)
    parser.add_argument("--batch_size", default=1, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)