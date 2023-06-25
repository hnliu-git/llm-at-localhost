
import time
import gradio as gr

from llm_at_localhost.models import *
from llm_at_localhost.utils import get_mem_utilization

def call_models(model_name, is_quantitized, text, length, temp):

    start_time = time.time()

    if model_name in ['EleutherAI/gpt-j-6B', 'distilgpt2']:
        model = HFModel(model_name, is_quantitized)
    else:
        model = HFModel(model_name, is_quantitized) 
    
    mem_load = get_mem_utilization()
    gen_text = model(text, length, temp)

    mem_inf = get_mem_utilization()
    time_used = time.time() - start_time

    return gen_text, time_used, mem_load, mem_inf


if __name__ == '__main__':

    with gr.Blocks() as demo:
        with gr.Row():
            models = gr.Radio(
                choices=["EleutherAI/gpt-j-6B", "distilgpt2", "none"],
                label="Choose the model"
            )
            is_quantitized = gr.Checkbox(label="Quantitized?")

        with gr.Row():
            temp = gr.Slider(step=1, label='temperature')
            length = gr.Slider(minimum=10, maximum=1000, step=10, label='length')

        text = gr.TextArea(label="Input")
        btn = gr.Button("Run")

        with gr.Row():
            time_used = gr.Textbox(label="Time:")
            mem_load = gr.Textbox(label="Mem for load:")
            mem_inf = gr.Textbox(label="Mem for inf")

        output = gr.TextArea(label="Output")

        btn.click(
            inputs=[models, is_quantitized, text, length, temp],
            outputs=[output, time_used, mem_load, mem_inf],
            fn=call_models
        )

    demo.launch()