from optimum.intel.openvino import OVWeightQuantizationConfig, OVModelForCausalLM
from transformers import AutoTokenizer
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
from optimum.intel.openvino import OVModelForCausalLM
import yaml
class OpenVINOLLMLoader:
    
    def __init__(self,logger):
        """
        Initialize the LLM loader using config.yaml.
        
        - logger: Python logging.Logger object for tracking and debugging.
        - export_path: Path of directory where openvino model will be exported to or loaded from.
        - hugging_face_model_name: Hugging Face model name or path
        - max_new_tokens: Max number of tokens to generate in a response
        - do_sample: Whether to use sampling during generation.
        - temperature: Controls randomness of model's output
        - top_p: Top-p nucleus sampling threshold
        """
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)["llm"]

        self.export_path = config["export_path"]
        self.hugging_face_model_name = config["hugging_face_model_name"]
        self.max_new_tokens = config["max_new_tokens"]
        self.do_sample = config["do_sample"]
        self.temperature = config["temperature"]
        self.top_p = config["top_p"]
        self.logger = logger

    def load_openvino_llm(self):
        """
        Load the quantized OpenVINO model and create a LangChain-compatible pipeline.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.export_path)
        model = OVModelForCausalLM.from_pretrained(self.export_path)

        text_gen_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id,
        )


        llm_model = HuggingFacePipeline(pipeline=text_gen_pipeline)

        return llm_model


    def get_llm_model(self):
        """
        Get the LLM model ready for inference. 
        If loading fails, it will download and quantize the model to INT4 using OpenVINO.
        """

        try:
            llm = self.load_openvino_llm()
            print(f"MODEL: Using already converted model at {self.export_path}")
            self.logger.info(f"Successfully loaded model from {self.export_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model from {self.export_path}: {e}")
            self.logger.info(f"Downloading and Converting the {self.hugging_face_model_name} Model to OVIR INT4 from scratch")

            q_config = OVWeightQuantizationConfig(bits=4, sym=True, group_size=128)
            model = OVModelForCausalLM.from_pretrained(self.hugging_face_model_name, export=True, quantization_config=q_config)
            model.save_pretrained(self.export_path)
            tokenizer = AutoTokenizer.from_pretrained(self.hugging_face_model_name)
            tokenizer.save_pretrained(self.export_path)
            llm = self.load_openvino_llm()
            self.logger.info(f"Successfully loaded model from {self.export_path}")

        return llm

if __name__ == "__main__":
    from log_setup import setup_logger
    logger = setup_logger(__name__, '')
    logger.info(f"-----------------------STARTED LOGGING---------------------------")
    loader = OpenVINOLLMLoader(logger)
    llm_model = loader.get_llm_model()