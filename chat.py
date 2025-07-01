import os
import logging
from typing import List, Dict, Any, Tuple
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationalAI:
    def __init__(self, model_path: str = "baidu/ERNIE-4.5-0.3B-PT"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.generation_config = {
            "max_new_tokens": 32768,
            "temperature": 0.8,
            "top_p": 0.95,
        }
        self._initialize_model()

    def _initialize_model(self) -> None:
        # Check for Apple Silicon MPS support
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple Silicon MPS acceleration")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using CUDA acceleration")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU (no acceleration available)")
        
        self.device = device
        
        logger.info("Initializing tokenizer components...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        
        logger.info("Loading neural network model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if device.type != "cpu" else torch.float32,
            trust_remote_code=True,
        ).to(device)
        logger.info("AI model initialization complete.")

    def _build_conversation_context(self, conversation_log: List[Dict], query: str) -> str:
        system_directive = {
            "role": "system", 
            "content": "You are an intelligent conversational agent. Provide responses exclusively in English."
        }
        
        message_sequence = [system_directive] + conversation_log + [{"role": "user", "content": query}]
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            message_sequence,
            tokenize=False,
            add_generation_prompt=True
        )
        return formatted_prompt

    def _calculate_throughput_metrics(self, token_count: int, elapsed_time: float) -> str:
        throughput = round(token_count / elapsed_time, 2) if elapsed_time > 0 else 0
        return f"⚡ {throughput} tok/s"

    def process_user_input(self, conversation_log: List[Dict], user_query: str) -> Tuple[List[Dict], str, str]:
        conversation_prompt = self._build_conversation_context(conversation_log, user_query)
        
        model_inputs = self.tokenizer([conversation_prompt], return_tensors="pt").to(self.device)
        
        inference_start = datetime.now()
        
        generated_output = self.model.generate(
            model_inputs.input_ids,
            **self.generation_config,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        inference_end = datetime.now()
        processing_duration = (inference_end - inference_start).total_seconds()
        
        new_token_ids = generated_output[0][model_inputs.input_ids.shape[-1]:]
        assistant_response = self.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        
        throughput_info = self._calculate_throughput_metrics(len(new_token_ids), processing_duration)
        
        updated_conversation = conversation_log + [
            {"role": "user", "content": user_query}, 
            {"role": "assistant", "content": assistant_response}
        ]
        
        return updated_conversation, "", throughput_info


def create_web_interface():
    ai_assistant = ConversationalAI()
    
    custom_theme = gr.themes.Monochrome(
        primary_hue="slate",
        secondary_hue="zinc",
        neutral_hue="stone",
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono")
    ).set(
        body_background_fill="*neutral_50",
        block_background_fill="white",
        block_border_width="1px",
        block_shadow="*shadow_drop_lg",
        button_primary_background_fill="*primary_600",
        button_primary_background_fill_hover="*primary_700"
    )
    
    interface = gr.Blocks(
        title="Neural Conversation Engine", 
        theme=custom_theme,
        css="""
        .gradio-container {
            max-width: 1200px !important;
            margin: auto;
            padding: 2rem;
        }
        .chat-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 16px;
            padding: 1rem;
        }
        .metrics-box {
            background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
            border-radius: 12px;
            padding: 1rem;
            color: white;
        }
        """
    )
    
    with interface:
        with gr.Column(elem_classes=["main-container"]):
            gr.HTML("""
            <div style="text-align: center; padding: 2rem 0;">
                <h1 style="background: linear-gradient(45deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3rem; font-weight: 800; margin: 0;">🤖 AI Chat</h1>
                <p style="color: #64748b; font-size: 1.2rem; margin-top: 0.5rem;">AI Conversation Platform</p>
            </div>
            """)
            
            with gr.Row(equal_height=True):
                with gr.Column(scale=4, elem_classes=["chat-container"]):
                    conversation_display = gr.Chatbot(
                        label="💬 Conversation Feed",
                        height=500,
                        type="messages",
                        show_label=True,
                        container=True,
                        bubble_full_width=False,
                        avatar_images=None
                    )
                
                with gr.Column(scale=1, elem_classes=["metrics-box"]):
                    gr.Markdown("### 📊 Performance")
                    metrics_panel = gr.Textbox(
                        label="Throughput Metrics",
                        interactive=False,
                        show_label=False,
                        container=False,
                        lines=3
                    )
            
            with gr.Row():
                with gr.Column(scale=5):
                    user_input_field = gr.Textbox(
                        placeholder="💭 Share your thoughts or ask a question...",
                        label="Your Message",
                        autofocus=True,
                        show_label=False,
                        container=False,
                        lines=2
                    )
                
                with gr.Column(scale=1, min_width=100):
                    send_button = gr.Button(
                        "🚀 Send",
                        variant="primary",
                        size="lg"
                    )
        
        def handle_input(history, message):
            return ai_assistant.process_user_input(history, message)
        
        user_input_field.submit(
            handle_input,
            [conversation_display, user_input_field], 
            [conversation_display, user_input_field, metrics_panel]
        )
        
        send_button.click(
            handle_input,
            [conversation_display, user_input_field], 
            [conversation_display, user_input_field, metrics_panel]
        )
    
    return interface


def main():
    web_app = create_web_interface()
    web_app.launch(server_port=9999)


if __name__ == "__main__":
    main()
