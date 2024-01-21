import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from item_scorers.item_scorer import ItemScorer
import torch.nn.functional as F
# import gc

class MNLIScorer(ItemScorer):

    def __init__(self, config):
        super().__init__(config)
        
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(config['item_scoring']['mnli_model'])
        self.nli_tokenizer = AutoTokenizer.from_pretrained(config['item_scoring']['mnli_model'])

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
           self.device = torch.device("cuda")
        #    torch.cuda.empty_cache()
        #    gc.collect()
           print("Using CUDA")
        
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS")
        self.nli_model.to(self.device)

    def score_items(self,preference,items) -> dict:
        
        #Anton[jan 15]: 2'nd level gc (first is at model loading)
        #if self.device == torch.device("cuda"):
        #   print('clearing CUDA cache and running gc')
        #   torch.cuda.empty_cache()
        #   gc.collect()
        
        # Read batch_size from config
        batch_size = int(self.config['item_scoring']['batch_size']) if 'batch_size' in self.config['item_scoring'] else len(items)

        like_probs = {item_id : None for item_id in items}
        item_ids = list(items.keys())

        for i in range(0, len(items), batch_size):
            batch_item_ids = item_ids[i:i + batch_size]
            batch_items = [items[item_id]['description'] for item_id in batch_item_ids]

            hypotheses = [preference] * batch_size
            premises = batch_items

            inputs = self.nli_tokenizer(premises, hypotheses, padding=True, truncation=True, return_tensors="pt", max_length=self.nli_tokenizer.model_max_length)

            # Ensure the tokenized inputs don't exceed the max length
            if inputs.input_ids.size(1) > self.nli_tokenizer.model_max_length:
                print(f"Warning: One or more tokenized inputs exceed the max length of {self.tokenizer.model_max_length} and will be truncated.")

            inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.nli_model(**inputs)
                entail_contradiction_logits = outputs[0][:,[0,2]]
                # predictions = entail_contradiction_logits.softmax(dim=1)
                T = self.config['item_scoring']['mnli_temp']
                if T == 0:
                    unrounded_preds = F.softmax(entail_contradiction_logits, dim=1)
                    predictions = torch.round(unrounded_preds)
                else:
                    predictions = F.softmax(entail_contradiction_logits/T, dim=1)

            #get entailement probs
            entailment_probs = predictions[:, 1].tolist()

            for j, item_id in enumerate(batch_item_ids):
                like_probs[item_id] = entailment_probs[j]

            #Anton[jan 15]: 3'd level gc
            #if self.device == torch.device("cuda"):
            #    print('clearing CUDA cache and running gc')
            #    torch.cuda.empty_cache()
            #    gc.collect()

        return like_probs