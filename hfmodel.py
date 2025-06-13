### modified from https://github.com/qiuqiangkong/audioset_tagging_cnn
import torch
from audioset_tagging_cnn.models import *
from audioset_tagging_cnn.config import sample_rate,classes_num,labels
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub import ModelCard
from torchinfo import summary

def load_panns_model(checkpoint_path,window_size=1024,hop_size=320,mel_bins=64,fmin=50,fmax=14000,model_type="ResNet22"):
    """Inference audio tagging result of an audio clip.
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    global sample_rate
    global classes_num
    global labels

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    model.eval()

    return model


class PANNS_Model(
        nn.Module,
        PyTorchModelHubMixin, # multiple inheritance
        library_name="pytorch",
        tags=["panns", "audio", "tagging"],
        license="apache-2.0",
        # ^ optional metadata to generate model card
        
        repo_url="https://github.com/qiuqiangkong/audioset_tagging_cnn",
        docs_url="https://github.com/qiuqiangkong/audioset_tagging_cnn",
        # ^ optional metadata to generate model card
    ):
    def __init__(self, checkpoint_path="ResNet22_mAP=0.430.pth",model_type="ResNet22"):
        super().__init__()
        
        self.backbone = load_panns_model(
            checkpoint_path=checkpoint_path,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            model_type=model_type
        )

        
    def forward(self, x):
        return self.backbone(x)

def inference(model,waveform, usecuda=False):
    """Inference audio tagging result of an audio clip.
    """
    device = torch.device('cuda') if usecuda and torch.cuda.is_available() else torch.device('cpu')

    if 'cuda' in str(device):
        model.to(device)
        waveform = waveform.to(device)

    # Forward
    with torch.no_grad():
        batch_output_dict = model(waveform)

    clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()
    sorted_indexes = np.argsort(clipwise_output)[::-1]

    if 'embedding' in batch_output_dict.keys():
        embedding = batch_output_dict['embedding'].data.cpu().numpy()
        
    return clipwise_output, labels, sorted_indexes, embedding

## add the __main__ function to run the model and save it
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='PANNS Model Conversion')
    parser.add_argument('--checkpoint_path', default='ResNet22_mAP=0.430.pth', type=str, help='Path to the model checkpoint')
    parser.add_argument('--model_type', default='ResNet22', type=str, help='Type of the model (e.g., ResNet22, MobileNetV2)')
    args = parser.parse_args()

    model = PANNS_Model(checkpoint_path="MobileNetV2_mAP=0.383.pth",model_type="MobileNetV2")

    print(summary(model)) 

    model.save_pretrained("panns_MobileNetV2")

    model.push_to_hub("nicofarr/panns_MobileNetV2")

    model = PANNS_Model.from_pretrained("nicofarr/panns_MobileNetV2")
    print(summary(model))

    card = ModelCard.load("nicofarr/panns_MobileNetV2")
    print(card.data.tags)
    print(card.data.library_name)