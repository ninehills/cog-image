from typing import List

import torch
from cog import BaseModel, BasePredictor, Input
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer


class Output(BaseModel):
    """Output schema for the predictor"""
    scores: List[float]


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        model_name_or_path = "checkpoints"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = ORTModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.model.to("cuda")

    def predict(self,
        pairs_json: str = Input(description='Input pairs, eg: [["a", "b"], ["c", "d"]]'),
    ) -> Output:
        """Run a single prediction on the model"""
        pairs = json.loads(pairs_json)
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
        return Output(scores=scores)
