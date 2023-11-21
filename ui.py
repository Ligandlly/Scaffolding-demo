import os
import gradio as gr
from openai import OpenAI

client = OpenAI()

import tomllib

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

messages = [{"role": "system", "content": config["system-prompt"]}]
text_display = []

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message

def main_note(filepath, history):
    if not filepath:
        return history
    print(filepath)
    audio_file = open(filepath, "rb")
    transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)

    user_input = transcript.text
    print(user_input)
    text_display.append(user_input)

    # GPT生成回答
    output_message = get_completion(user_input)

    messages.append(output_message)

    output_text = output_message.content
    print(output_text)
    speech_file_path = "/tmp/speech.wav"
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=output_text,
    )

    response.stream_to_file(speech_file_path)

    # Ref: https://github.com/gradio-app/gradio/issues/2768#issuecomment-1497976532
    history.append(((filepath,), (speech_file_path,)))

    text_display.append(output_text)
    return history, "\n".join(text_display)


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    mic = gr.Audio(sources=["microphone"], type="filepath")
    text = gr.Textbox()
    mic.change(main_note, [mic, chatbot], [chatbot, text])


demo.launch(share=True)
