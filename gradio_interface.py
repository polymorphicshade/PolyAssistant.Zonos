import os
import torch
import torchaudio
import gradio as gr
from os import getenv
from zonos.model import Zonos, DEFAULT_BACKBONE_CLS as ZonosBackbone
from zonos.conditioning import make_cond_dict, supported_language_codes
from zonos.utils import DEFAULT_DEVICE as device
import io
from flask import Flask, request, jsonify, Response
import threading
import tempfile

CURRENT_MODEL_TYPE = None
CURRENT_MODEL = None

SPEAKER_EMBEDDING = None
SPEAKER_AUDIO_PATH = None

app = Flask(__name__)

def load_model_if_needed(model_choice: str):
    global CURRENT_MODEL_TYPE, CURRENT_MODEL
    if CURRENT_MODEL_TYPE != model_choice:
        if CURRENT_MODEL is not None:
            del CURRENT_MODEL
            torch.cuda.empty_cache()
        print(f"Loading {model_choice} model...")
        CURRENT_MODEL = Zonos.from_pretrained(model_choice, device=device)
        CURRENT_MODEL.requires_grad_(False).eval()
        CURRENT_MODEL_TYPE = model_choice
        print(f"{model_choice} model loaded successfully!")
    return CURRENT_MODEL


def update_ui(model_choice):
    """
    Dynamically show/hide UI elements based on the model's conditioners.
    We do NOT display 'language_id' or 'ctc_loss' even if they exist in the model.
    """
    model = load_model_if_needed(model_choice)
    cond_names = [c.name for c in model.prefix_conditioner.conditioners]
    print("Conditioners in this model:", cond_names)

    text_update = gr.update(visible=("espeak" in cond_names))
    language_update = gr.update(visible=("espeak" in cond_names))
    speaker_audio_update = gr.update(visible=("speaker" in cond_names))
    prefix_audio_update = gr.update(visible=True)
    emotion1_update = gr.update(visible=("emotion" in cond_names))
    emotion2_update = gr.update(visible=("emotion" in cond_names))
    emotion3_update = gr.update(visible=("emotion" in cond_names))
    emotion4_update = gr.update(visible=("emotion" in cond_names))
    emotion5_update = gr.update(visible=("emotion" in cond_names))
    emotion6_update = gr.update(visible=("emotion" in cond_names))
    emotion7_update = gr.update(visible=("emotion" in cond_names))
    emotion8_update = gr.update(visible=("emotion" in cond_names))
    vq_single_slider_update = gr.update(visible=("vqscore_8" in cond_names))
    fmax_slider_update = gr.update(visible=("fmax" in cond_names))
    pitch_std_slider_update = gr.update(visible=("pitch_std" in cond_names))
    speaking_rate_slider_update = gr.update(visible=("speaking_rate" in cond_names))
    dnsmos_slider_update = gr.update(visible=("dnsmos_ovrl" in cond_names))
    speaker_noised_checkbox_update = gr.update(visible=("speaker_noised" in cond_names))
    unconditional_keys_update = gr.update(
        choices=[name for name in cond_names if name not in ("espeak", "language_id")]
    )

    return (
        text_update,
        language_update,
        speaker_audio_update,
        prefix_audio_update,
        emotion1_update,
        emotion2_update,
        emotion3_update,
        emotion4_update,
        emotion5_update,
        emotion6_update,
        emotion7_update,
        emotion8_update,
        vq_single_slider_update,
        fmax_slider_update,
        pitch_std_slider_update,
        speaking_rate_slider_update,
        dnsmos_slider_update,
        speaker_noised_checkbox_update,
        unconditional_keys_update,
    )


def generate_audio(
    model_choice,
    text,
    language,
    speaker_audio,
    prefix_audio,
    e1,
    e2,
    e3,
    e4,
    e5,
    e6,
    e7,
    e8,
    vq_single,
    fmax,
    pitch_std,
    speaking_rate,
    dnsmos_ovrl,
    speaker_noised,
    cfg_scale,
    top_p,
    top_k,
    min_p,
    linear,
    confidence,
    quadratic,
    seed,
    randomize_seed,
    unconditional_keys,
    progress=gr.Progress(),
):
    """
    Generates audio based on the provided UI parameters.
    We do NOT use language_id or ctc_loss even if the model has them.
    """
    selected_model = load_model_if_needed(model_choice)

    speaker_noised_bool = bool(speaker_noised)
    fmax = float(fmax)
    pitch_std = float(pitch_std)
    speaking_rate = float(speaking_rate)
    dnsmos_ovrl = float(dnsmos_ovrl)
    cfg_scale = float(cfg_scale)
    top_p = float(top_p)
    top_k = int(top_k)
    min_p = float(min_p)
    linear = float(linear)
    confidence = float(confidence)
    quadratic = float(quadratic)
    seed = int(seed)
    max_new_tokens = 86 * 30

    # This is a bit ew, but works for now.
    global SPEAKER_AUDIO_PATH, SPEAKER_EMBEDDING

    if randomize_seed:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    torch.manual_seed(seed)

    if speaker_audio is not None and "speaker" not in unconditional_keys:
        if speaker_audio != SPEAKER_AUDIO_PATH:
            print("Recomputed speaker embedding")
            wav, sr = torchaudio.load(speaker_audio)
            SPEAKER_EMBEDDING = selected_model.make_speaker_embedding(wav, sr)
            SPEAKER_EMBEDDING = SPEAKER_EMBEDDING.to(device, dtype=torch.bfloat16)
            SPEAKER_AUDIO_PATH = speaker_audio

    audio_prefix_codes = None
    if prefix_audio is not None:
        wav_prefix, sr_prefix = torchaudio.load(prefix_audio)
        wav_prefix = wav_prefix.mean(0, keepdim=True)
        wav_prefix = selected_model.autoencoder.preprocess(wav_prefix, sr_prefix)
        wav_prefix = wav_prefix.to(device, dtype=torch.float32)
        audio_prefix_codes = selected_model.autoencoder.encode(wav_prefix.unsqueeze(0))

    emotion_tensor = torch.tensor(list(map(float, [e1, e2, e3, e4, e5, e6, e7, e8])), device=device)

    vq_val = float(vq_single)
    vq_tensor = torch.tensor([vq_val] * 8, device=device).unsqueeze(0)

    cond_dict = make_cond_dict(
        text=text,
        language=language,
        speaker=SPEAKER_EMBEDDING,
        emotion=emotion_tensor,
        vqscore_8=vq_tensor,
        fmax=fmax,
        pitch_std=pitch_std,
        speaking_rate=speaking_rate,
        dnsmos_ovrl=dnsmos_ovrl,
        speaker_noised=speaker_noised_bool,
        device=device,
        unconditional_keys=unconditional_keys,
    )
    conditioning = selected_model.prepare_conditioning(cond_dict)

    estimated_generation_duration = 30 * len(text) / 400
    estimated_total_steps = int(estimated_generation_duration * 86)

    def update_progress(_frame: torch.Tensor, step: int, _total_steps: int) -> bool:
        progress((step, estimated_total_steps))
        return True

    codes = selected_model.generate(
        prefix_conditioning=conditioning,
        audio_prefix_codes=audio_prefix_codes,
        max_new_tokens=max_new_tokens,
        cfg_scale=cfg_scale,
        batch_size=1,
        sampling_params=dict(top_p=top_p, top_k=top_k, min_p=min_p, linear=linear, conf=confidence, quad=quadratic),
        callback=update_progress,
    )

    wav_out = selected_model.autoencoder.decode(codes).cpu().detach()
    sr_out = selected_model.autoencoder.sampling_rate
    if wav_out.dim() == 2 and wav_out.size(0) > 1:
        wav_out = wav_out[0:1, :]
    return (sr_out, wav_out.squeeze().numpy()), seed

def generate_audio_api(
    model_choice="Zyphra/Zonos-v0.1-transformer",
    text="Zonos uses eSpeak for text to phoneme conversion!",
    language="en-us",
    speaker_audio=None,
    prefix_audio=None,
    e1=1.0,
    e2=0.05,
    e3=0.05,
    e4=0.05,
    e5=0.05,
    e6=0.05,
    e7=0.1,
    e8=0.2,
    vq_single=0.78,
    fmax=24000,
    pitch_std=45.0,
    speaking_rate=15.0,
    dnsmos_ovrl=4.0,
    speaker_noised=False,
    cfg_scale=2.0,
    top_p=0.0,
    top_k=0,
    min_p=0.0,
    linear=0.5,
    confidence=0.4,
    quadratic=0.0,
    seed=420,
    randomize_seed=True,
    unconditional_keys=["emotion"],
):
    speaker_audio_path = None
    if speaker_audio and hasattr(speaker_audio, 'read'):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(speaker_audio.read())
            speaker_audio_path = tmp.name
    
    prefix_audio_path = None
    if prefix_audio and hasattr(prefix_audio, 'read'):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(prefix_audio.read())
            prefix_audio_path = tmp.name
    
    (sr, audio_data), _ = generate_audio(
        model_choice=model_choice,
        text=text,
        language=language,
        speaker_audio=speaker_audio_path,
        prefix_audio=prefix_audio_path,
        e1=e1,
        e2=e2,
        e3=e3,
        e4=e4,
        e5=e5,
        e6=e6,
        e7=e7,
        e8=e8,
        vq_single=vq_single,
        fmax=fmax,
        pitch_std=pitch_std,
        speaking_rate=speaking_rate,
        dnsmos_ovrl=dnsmos_ovrl,
        speaker_noised=speaker_noised,
        cfg_scale=cfg_scale,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        linear=linear,
        confidence=confidence,
        quadratic=quadratic,
        seed=seed,
        randomize_seed=randomize_seed,
        unconditional_keys=unconditional_keys,
    )
    
    if speaker_audio_path:
        os.unlink(speaker_audio_path)
    if prefix_audio_path:
        os.unlink(prefix_audio_path)
    
    buffer = io.BytesIO()
    torchaudio.save(buffer, torch.from_numpy(audio_data).unsqueeze(0), sr, format='wav')
    buffer.seek(0)
    
    return buffer.getvalue()

@app.route('/api/generate_audio', methods=['POST'])
def api_generate_audio():
    try:
        data = request.get_json()
        
        params = {
            'model_choice': data.get('model_choice', "Zyphra/Zonos-v0.1-transformer"),
            'text': data.get('text', "Zonos uses eSpeak for text to phoneme conversion!"),
            'language': data.get('language', "en-us"),
            'speaker_audio': data.get('speaker_audio'),
            'prefix_audio': data.get('prefix_audio', "assets/silence_100ms.wav"),
            'e1': data.get('e1', 1.0),
            'e2': data.get('e2', 0.05),
            'e3': data.get('e3', 0.05),
            'e4': data.get('e4', 0.05),
            'e5': data.get('e5', 0.05),
            'e6': data.get('e6', 0.05),
            'e7': data.get('e7', 0.1),
            'e8': data.get('e8', 0.2),
            'vq_single': data.get('vq_single', 0.78),
            'fmax': data.get('fmax', 24000),
            'pitch_std': data.get('pitch_std', 45.0),
            'speaking_rate': data.get('speaking_rate', 15.0),
            'dnsmos_ovrl': data.get('dnsmos_ovrl', 4.0),
            'speaker_noised': data.get('speaker_noised', False),
            'cfg_scale': data.get('cfg_scale', 2.0),
            'top_p': data.get('top_p', 0.0),
            'top_k': data.get('top_k', 0),
            'min_p': data.get('min_p', 0.0),
            'linear': data.get('linear', 0.5),
            'confidence': data.get('confidence', 0.4),
            'quadratic': data.get('quadratic', 0.0),
            'seed': data.get('seed', 420),
            'randomize_seed': data.get('randomize_seed', True),
            'unconditional_keys': data.get('unconditional_keys', ["emotion"]),
        }
        
        (sr, audio_data), seed = generate_audio(**params)
        
        audio_list = audio_data.tolist()
        
        return jsonify({
            'status': 'success',
            'sample_rate': sr,
            'audio_data': audio_list,
            'seed': seed
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/generate_audio_raw', methods=['POST'])
def api_generate_audio_raw():
    try:
        model_choice = request.form.get('model_choice', "Zyphra/Zonos-v0.1-transformer")
        text = request.form.get('text', "Zonos uses eSpeak for text to phoneme conversion!")
        language = request.form.get('language', "en-us")
        
        speaker_audio = request.files.get('speaker_audio')
        prefix_audio = request.files.get('prefix_audio')
        
        e1 = float(request.form.get('e1', 1.0))
        e2 = float(request.form.get('e2', 0.05))
        e3 = float(request.form.get('e3', 0.05))
        e4 = float(request.form.get('e4', 0.05))
        e5 = float(request.form.get('e5', 0.05))
        e6 = float(request.form.get('e6', 0.05))
        e7 = float(request.form.get('e7', 0.1))
        e8 = float(request.form.get('e8', 0.2))
        vq_single = float(request.form.get('vq_single', 0.78))
        fmax = float(request.form.get('fmax', 24000))
        pitch_std = float(request.form.get('pitch_std', 45.0))
        speaking_rate = float(request.form.get('speaking_rate', 15.0))
        dnsmos_ovrl = float(request.form.get('dnsmos_ovrl', 4.0))
        speaker_noised = request.form.get('speaker_noised', 'false').lower() in ('true', '1', 't')
        cfg_scale = float(request.form.get('cfg_scale', 2.0))
        top_p = float(request.form.get('top_p', 0.0))
        top_k = int(request.form.get('top_k', 0))
        min_p = float(request.form.get('min_p', 0.0))
        linear = float(request.form.get('linear', 0.5))
        confidence = float(request.form.get('confidence', 0.4))
        quadratic = float(request.form.get('quadratic', 0.0))
        seed = int(request.form.get('seed', 420))
        randomize_seed = request.form.get('randomize_seed', 'true').lower() in ('true', '1', 't')
        
        unconditional_keys_str = request.form.get('unconditional_keys', "emotion")
        unconditional_keys = [k.strip() for k in unconditional_keys_str.split(',') if k.strip()]
        
        wav_data = generate_audio_api(
            model_choice=model_choice,
            text=text,
            language=language,
            speaker_audio=speaker_audio,
            prefix_audio=prefix_audio,
            e1=e1,
            e2=e2,
            e3=e3,
            e4=e4,
            e5=e5,
            e6=e6,
            e7=e7,
            e8=e8,
            vq_single=vq_single,
            fmax=fmax,
            pitch_std=pitch_std,
            speaking_rate=speaking_rate,
            dnsmos_ovrl=dnsmos_ovrl,
            speaker_noised=speaker_noised,
            cfg_scale=cfg_scale,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            linear=linear,
            confidence=confidence,
            quadratic=quadratic,
            seed=seed,
            randomize_seed=randomize_seed,
            unconditional_keys=unconditional_keys,
        )
        
        return Response(
            wav_data,
            mimetype='audio/wav',
            headers={
                'Content-Disposition': 'attachment; filename=generated_audio.wav'
            }
        )
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def build_interface():
    supported_models = []
    if "transformer" in ZonosBackbone.supported_architectures:
        supported_models.append("Zyphra/Zonos-v0.1-transformer")

    if "hybrid" in ZonosBackbone.supported_architectures:
        supported_models.append("Zyphra/Zonos-v0.1-hybrid")
    else:
        print(
            "| The current ZonosBackbone does not support the hybrid architecture, meaning only the transformer model will be available in the model selector.\n"
            "| This probably means the mamba-ssm library has not been installed."
        )

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                model_choice = gr.Dropdown(
                    choices=supported_models,
                    value=supported_models[0],
                    label="Zonos Model Type",
                    info="Select the model variant to use.",
                )
                text = gr.Textbox(
                    label="Text to Synthesize",
                    value="Zonos uses eSpeak for text to phoneme conversion!",
                    lines=4,
                    max_length=500,  # approximately
                )
                language = gr.Dropdown(
                    choices=supported_language_codes,
                    value="en-us",
                    label="Language Code",
                    info="Select a language code.",
                )
            prefix_audio = gr.Audio(
                value="assets/silence_100ms.wav",
                label="Optional Prefix Audio (continue from this audio)",
                type="filepath",
            )
            with gr.Column():
                speaker_audio = gr.Audio(
                    label="Optional Speaker Audio (for cloning)",
                    type="filepath",
                )
                speaker_noised_checkbox = gr.Checkbox(label="Denoise Speaker?", value=False)

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Conditioning Parameters")
                dnsmos_slider = gr.Slider(1.0, 5.0, value=4.0, step=0.1, label="DNSMOS Overall")
                fmax_slider = gr.Slider(0, 24000, value=24000, step=1, label="Fmax (Hz)")
                vq_single_slider = gr.Slider(0.5, 0.8, 0.78, 0.01, label="VQ Score")
                pitch_std_slider = gr.Slider(0.0, 300.0, value=45.0, step=1, label="Pitch Std")
                speaking_rate_slider = gr.Slider(5.0, 30.0, value=15.0, step=0.5, label="Speaking Rate")

            with gr.Column():
                gr.Markdown("## Generation Parameters")
                cfg_scale_slider = gr.Slider(1.0, 5.0, 2.0, 0.1, label="CFG Scale")
                seed_number = gr.Number(label="Seed", value=420, precision=0)
                randomize_seed_toggle = gr.Checkbox(label="Randomize Seed (before generation)", value=True)

        with gr.Accordion("Sampling", open=False):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### NovelAi's unified sampler")
                    linear_slider = gr.Slider(-2.0, 2.0, 0.5, 0.01, label="Linear (set to 0 to disable unified sampling)", info="High values make the output less random.")
                    #Conf's theoretical range is between -2 * Quad and 0.
                    confidence_slider = gr.Slider(-2.0, 2.0, 0.40, 0.01, label="Confidence", info="Low values make random outputs more random.")
                    quadratic_slider = gr.Slider(-2.0, 2.0, 0.00, 0.01, label="Quadratic", info="High values make low probablities much lower.")
                with gr.Column():
                    gr.Markdown("### Legacy sampling")
                    top_p_slider = gr.Slider(0.0, 1.0, 0, 0.01, label="Top P")
                    min_k_slider = gr.Slider(0.0, 1024, 0, 1, label="Min K")
                    min_p_slider = gr.Slider(0.0, 1.0, 0, 0.01, label="Min P")

        with gr.Accordion("Advanced Parameters", open=False):
            gr.Markdown(
                "### Unconditional Toggles\n"
                "Checking a box will make the model ignore the corresponding conditioning value and make it unconditional.\n"
                'Practically this means the given conditioning feature will be unconstrained and "filled in automatically".'
            )
            with gr.Row():
                unconditional_keys = gr.CheckboxGroup(
                    [
                        "speaker",
                        "emotion",
                        "vqscore_8",
                        "fmax",
                        "pitch_std",
                        "speaking_rate",
                        "dnsmos_ovrl",
                        "speaker_noised",
                    ],
                    value=["emotion"],
                    label="Unconditional Keys",
                )

            gr.Markdown(
                "### Emotion Sliders\n"
                "Warning: The way these sliders work is not intuitive and may require some trial and error to get the desired effect.\n"
                "Certain configurations can cause the model to become unstable. Setting emotion to unconditional may help."
            )
            with gr.Row():
                emotion1 = gr.Slider(0.0, 1.0, 1.0, 0.05, label="Happiness")
                emotion2 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Sadness")
                emotion3 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Disgust")
                emotion4 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Fear")
            with gr.Row():
                emotion5 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Surprise")
                emotion6 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Anger")
                emotion7 = gr.Slider(0.0, 1.0, 0.1, 0.05, label="Other")
                emotion8 = gr.Slider(0.0, 1.0, 0.2, 0.05, label="Neutral")

        with gr.Column():
            generate_button = gr.Button("Generate Audio")
            output_audio = gr.Audio(label="Generated Audio", type="numpy", autoplay=True)

        model_choice.change(
            fn=update_ui,
            inputs=[model_choice],
            outputs=[
                text,
                language,
                speaker_audio,
                prefix_audio,
                emotion1,
                emotion2,
                emotion3,
                emotion4,
                emotion5,
                emotion6,
                emotion7,
                emotion8,
                vq_single_slider,
                fmax_slider,
                pitch_std_slider,
                speaking_rate_slider,
                dnsmos_slider,
                speaker_noised_checkbox,
                unconditional_keys,
            ],
        )

        # On page load, trigger the same UI refresh
        demo.load(
            fn=update_ui,
            inputs=[model_choice],
            outputs=[
                text,
                language,
                speaker_audio,
                prefix_audio,
                emotion1,
                emotion2,
                emotion3,
                emotion4,
                emotion5,
                emotion6,
                emotion7,
                emotion8,
                vq_single_slider,
                fmax_slider,
                pitch_std_slider,
                speaking_rate_slider,
                dnsmos_slider,
                speaker_noised_checkbox,
                unconditional_keys,
            ],
        )

        # Generate audio on button click
        generate_button.click(
            fn=generate_audio,
            inputs=[
                model_choice,
                text,
                language,
                speaker_audio,
                prefix_audio,
                emotion1,
                emotion2,
                emotion3,
                emotion4,
                emotion5,
                emotion6,
                emotion7,
                emotion8,
                vq_single_slider,
                fmax_slider,
                pitch_std_slider,
                speaking_rate_slider,
                dnsmos_slider,
                speaker_noised_checkbox,
                cfg_scale_slider,
                top_p_slider,
                min_k_slider,
                min_p_slider,
                linear_slider,
                confidence_slider,
                quadratic_slider,
                seed_number,
                randomize_seed_toggle,
                unconditional_keys,
            ],
            outputs=[output_audio, seed_number],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    share = getenv("GRADIO_SHARE", "False").lower() in ("true", "1", "t")

    flask_thread = threading.Thread(target=app.run, kwargs={
        "host": "0.0.0.0",
        "port": 7861
    })
    flask_thread.daemon = True
    flask_thread.start()

    demo.launch(server_name="0.0.0.0", server_port=7860, share=share)
