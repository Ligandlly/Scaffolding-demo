import gradio as gr
from openai import OpenAI

client = OpenAI()

messages = [{"role": "system", "content": "You are a helpful assistant."}]


def main_note(filepath):
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
    speech_file_path = "/tmp/speech.mp3"
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=output_text,
    )

    response.stream_to_file(speech_file_path)
    return speech_file_path


demo = gr.Interface(
    fn=main_note,
    inputs=gr.Audio(sources=["microphone"], type="filepath", format="mp3"),
    outputs="audio",
)

if __name__ == "__main__":
    demo.launch(share=True)
