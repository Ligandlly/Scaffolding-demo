import os
import gradio as gr
from openai import OpenAI

client = OpenAI()

messages = [{"role": "system", "content": "You are a helpful assistant."}]


def main_note(filepath, history):
    if not filepath:
        return history
    print(filepath)
    audio_file = open(filepath, "rb")
    transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)

    user_input = transcript.text
    print(user_input)

    messages.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    output_message = response.choices[0].message
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

    os.rename(filepath, filepath + ".wav")
    os.rename(speech_file_path, speech_file_path + ".wav")
    history.append(((filepath + ".wav",), (speech_file_path + ".wav",)))
    return history


# def get_chatbot_response(x, history):
#     l = [(x, x)]
#     history += l
#     return history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()

    mic = gr.Audio(sources=["microphone"], type="filepath")
    mic.change(main_note, [mic, chatbot], chatbot)


demo.launch(share=True)
