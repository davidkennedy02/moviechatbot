from dialogue_downloader import downloadData
from dataset_generator import MovieDataset
from transformers import BartForConditionalGeneration, BartTokenizer
from fine_tuning import fineTuneModel

def chat(model: BartForConditionalGeneration, input_text:str):
    inputs = tokenizer(input_text, return_tensors="pt")
    
    outputs = model.generate(
        **inputs,
        max_length=50,
        num_beams=5,
        early_stopping=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def begin_conversation(model):
    while True:
        user_input=input("You: ")
        if user_input.lower() == "exit":
            break
        print(f"Bot: {chat(model, user_input)}")


if __name__ == "__main__":
    
    # downloadData()
    
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    
    # Generate dataset
    dataset = MovieDataset("movie_dialogues.csv", tokenizer=tokenizer)
    
    # Define model 
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    
    # Fine-tune model 
    fineTuneModel(model=model, dataset=dataset, tokenizer=tokenizer)
    
    # Begin conversation
    begin_conversation(model)