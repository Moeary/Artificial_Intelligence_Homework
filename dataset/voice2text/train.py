import whisper
import gradio as gr

# 加载Whisper模型
model = whisper.load_model("base")

# 定义语音识别函数
def transcribe(audio):
    # 使用Whisper模型进行语音识别，指定语言为简体中文
    result = model.transcribe(audio, language="zh")
    segments = result["segments"]
    
    # 构建SRT格式的输出
    srt_output = ""
    for i, segment in enumerate(segments):
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]
        
        # 格式化时间为SRT格式
        start_time_str = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02},{int((start_time % 1) * 1000):03}"
        end_time_str = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02},{int((end_time % 1) * 1000):03}"
        
        # 构建SRT条目
        srt_output += f"{i + 1}\n{start_time_str} --> {end_time_str}\n{text}\n\n"
    
    return srt_output

# Gradio界面
audio_input = gr.Audio(type="filepath", label="Speak or Upload Audio")
output_text = gr.Textbox(label="Transcribed Text (SRT Format)")

gr.Interface(fn=transcribe, inputs=audio_input, outputs=output_text, title="Speech to Text", description="Speak into the microphone or upload an audio file to transcribe it to text using Whisper. The output will be in SRT format.").launch()