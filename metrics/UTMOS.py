import torch
import utmosv2

"""
UTMOS score, automatic Mean Opinion Score (MOS) prediction system,
using UTMOSv2: https://github.com/sarulab-speech/UTMOSv2
"""


class UTMOSScore:
    """Predicting score for each audio clip using UTMOSv2."""

    def __init__(self, device, ckpt_path=None):
        """
        Initialize UTMOSv2 model.
        
        Args:
            device: torch device to use (e.g., 'cuda' or 'cpu')
            ckpt_path: optional custom checkpoint path (not used in new version)
        """
        self.device = device
        # Create pretrained UTMOSv2 model
        self.model = utmosv2.create_model(pretrained=True)
        self.model = self.model.to(device)
        self.model.eval()

    def score(self, wavs: torch.tensor) -> torch.tensor:
        """
        Predict MOS score for audio waveform(s).
        
        Args:
            wavs: audio waveform to be evaluated. When len(wavs) == 1 or 2,
                the model processes the input as a single audio clip. The model
                performs batch processing when len(wavs) == 3.
                
        Returns:
            torch.tensor: MOS scores scaled to 1-5 range
        """
        # Handle different input dimensions
        if len(wavs.shape) == 1:
            # Single waveform -> add batch and channel dims
            out_wavs = wavs.unsqueeze(0)
        elif len(wavs.shape) == 2:
            # Already has batch dimension
            out_wavs = wavs
        elif len(wavs.shape) == 3:
            # Full batch format
            out_wavs = wavs
        else:
            raise ValueError("Dimension of input tensor needs to be <= 3.")
        
        # Move to device
        out_wavs = out_wavs.to(self.device)
        
        # Predict using UTMOSv2
        # Assuming 16kHz sampling rate (UTMOSv2 default)
        with torch.no_grad():
            # UTMOSv2 expects numpy array or torch tensor
            # Returns predictions in 1-5 scale
            if out_wavs.is_cuda:
                predictions = self.model.predict(
                    data=out_wavs.cpu().numpy(), 
                    sr=16000
                )
            else:
                predictions = self.model.predict(
                    data=out_wavs.numpy(), 
                    sr=16000
                )
        
        # Convert predictions to tensor and ensure proper shape
        if isinstance(predictions, list):
            # Extract MOS values from prediction dicts if needed
            mos_values = torch.tensor([p.get('predicted_mos', p) for p in predictions])
        else:
            mos_values = torch.tensor(predictions)
        
        # Return in same format as original (detached CPU tensor)
        return mos_values.cpu().detach()