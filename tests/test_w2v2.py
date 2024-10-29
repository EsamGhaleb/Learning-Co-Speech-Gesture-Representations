import os
import tempfile
import torch
import torch.nn.functional as F
import shutil

from model.wav2vec2_wrapper import Wav2Vec2Wrapper, Wav2Vec2CNN
from model.peft_wrapper import get_lora_config

from pytorch_lightning import Trainer


def _get_dummy_audio(batch=8, freq=16000, length_samples=2):
    x = torch.rand((batch, 1, freq * length_samples))
    return x

def _get_all_parameters(model):
    total_params = sum(
        p.numel() for p in model.parameters()
    )
    return total_params

def _get_trainable_parameters(model):
    trainable_params = sum(
	    p.numel() for p in model.parameters() if p.requires_grad
    )

    return trainable_params


def test_wav2vec2_setup():
    for w2v2_type in ["base", "multilingual", "dutch_finetuned"]:
        try:
            _ = Wav2Vec2Wrapper(w2v2_type=w2v2_type)
        except:
            raise ImportError(f"Failed to initialize wav2vec2 wrapper with {w2v2_type} type")
        try:
            _ = Wav2Vec2CNN(length_samples=2, w2v2_type=w2v2_type)
        except:
            raise ImportError(f"Failed to initialize Wav2Vec2CNN with {w2v2_type} type")


def test_wav2vec2_forward_shape():
    batch_size = 8
    length_samples = 2
    x = _get_dummy_audio(batch_size, length_samples=length_samples)
    model = Wav2Vec2CNN(length_samples=length_samples)
    out = model(x)
    assert list(out.shape) == [batch_size, 128], "wrong output shape"


def test_wav2vec2_lora_setup():
    lora_config = get_lora_config()
    w2v2cnn = Wav2Vec2CNN(
        w2v2_type="dutch_finetuned",
        freeze=True,
    )

    num_param_w2v2cnn = _get_all_parameters(w2v2cnn)
    num_train_param_w2v2cnn = _get_trainable_parameters(w2v2cnn)

    try:
        peft_model = Wav2Vec2CNN(
            w2v2_type="dutch_finetuned",
            freeze=True,
            peft_config=lora_config
        )
    except:
        raise ImportError("Failed to initialize PEFT model with LoRA adapter")
    print(peft_model)

    assert (
        _get_all_parameters(peft_model) > num_param_w2v2cnn,
        "LoRA did not introduce new parameters"
    )
    assert (
        _get_trainable_parameters(peft_model) > num_train_param_w2v2cnn, 
        "LoRA did not introduce new trainable parameters"
    )


def test_wav2vec2_lora_forward():
    lora_config = get_lora_config()
    w2v2cnn_peft = Wav2Vec2CNN(
        w2v2_type="dutch_finetuned",
        freeze=True,
        peft_config=lora_config
    )

    w2v2cnn_no_peft =  Wav2Vec2CNN(
        w2v2_type="dutch_finetuned",
        freeze=True,
    )

    x = _get_dummy_audio(batch=2)
    peft_out = w2v2cnn_peft(x)
    nopeft_out = w2v2cnn_no_peft(x)

    assert peft_out.shape == nopeft_out.shape, "Wrong output shape for PEFT model"
    assert not torch.allclose(peft_out, nopeft_out), "Same outputs embeddings for PEFT and no-PEFT model"


def test_wav2vec2_lora_load_save():
    # temporary folder for checkpoint
    test_dir = tempfile.mkdtemp(dir='./')
    model_path = os.path.join(test_dir, "test_checkpoint.ckpt")

    # init peft model for wav2vec2cnn
    lora_config = get_lora_config()
    w2v2cnn_peft = Wav2Vec2CNN(
        w2v2_type="dutch_finetuned",
        freeze=True,
        peft_config=lora_config
    )

    # save checkpoints using Lightning Trainer
    trainer = Trainer(default_root_dir=test_dir)
    trainer.strategy.connect(w2v2cnn_peft)
    trainer.save_checkpoint(model_path)

    # init audio
    x = _get_dummy_audio(batch=2)
    
    # init peft model and load the weights from the checkpoint
    
    # Starting from lightning 2.2.0 load_from_checkpoint can be done directly
    # using class
    peft_from_checkpoint = Wav2Vec2CNN.load_from_checkpoint(model_path)
    print(peft_from_checkpoint)

    # If using an earlier version, it might be needed to load the checkpoint
    # from the instance as follows 
    # peft_from_checkpoint = Wav2Vec2CNN(
    #     w2v2_type="dutch_finetuned",
    #     freeze=True,
    #     peft_config=lora_config
    # )
    # peft_from_checkpoint.load_from_checkpoint(model_path)

    # Make sure ouputs of model before and after checkpointing are the same
    assert (
        torch.allclose(w2v2cnn_peft(x), peft_from_checkpoint(x)),
        "Loaded PEFT model from checkpoint is different"
    )

    shutil.rmtree(test_dir)


def test_lengths():
    import random
    batch_size = 16
    length_samples = 5
    sample_rate = 16000
    max_len_frames = length_samples * sample_rate
    model = Wav2Vec2CNN(length_samples=length_samples, w2v2_type="multilingual")
    model.eval()
    # Generate random lengths and tensors of that length
    lengths = [random.randint(max_len_frames - sample_rate, max_len_frames) for _ in range(batch_size)]
    speech = [torch.rand(1, lengths[i]) for i in range(batch_size)]

    # pad speech utterances with zeros till the required length
    pad = [(0, max_len_frames - lengths[i]) for i in range(batch_size)]
    for i in range(batch_size):
        curr_speech = speech[i]
        padded = F.pad(curr_speech, pad[i], "constant", 0)
        speech[i] = padded
    padded_speech = torch.stack(speech)
    assert tuple(padded_speech.shape) == (batch_size, 1, max_len_frames), "Padded speech shape mismatch"

    items = {}
    items["speech"] = {}
    items["speech"]["view1"] = padded_speech
    items["speech"]["lengths"] = torch.tensor(lengths)

    out_no_lengths = model(items["speech"]["view1"])
    out_lengths = model(items["speech"]["view1"], items["speech"]["lengths"])
    assert (
        not torch.allclose(out_no_lengths, out_lengths),
        "Lengths are not used: output does not change."
    )
    