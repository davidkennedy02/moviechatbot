from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from torch.utils.data import Subset

def fineTuneModel(model, dataset, tokenizer):
    
    # Calculate the number of samples to use (0.1% of the dataset)
    subset_size = max(1, len(dataset) // 1000)
    subset_indices = list(range(subset_size))

    # Create a new dataset containing only 0.1% of the original samples
    small_dataset = Subset(dataset, subset_indices)

    training_args = Seq2SeqTrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=16,
        learning_rate=5e-5,
        logging_steps=100,
        save_steps=500,
        eval_strategy="no",
        predict_with_generate=True,
        fp16=False
    )

    trainer = Seq2SeqTrainer(
        model=model, 
        args=training_args,
        train_dataset=small_dataset,
        processing_class=tokenizer
    )

    trainer.train()