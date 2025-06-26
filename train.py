import os
import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)
from PIL import Image
import evaluate
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load dataset
dataset = load_dataset("Hemg/new-plant-diseases-dataset")

# Create validation split if not present
if "validation" not in dataset:
    split = dataset["train"].train_test_split(test_size=0.2, seed=42)
    dataset = DatasetDict({
        "train": split["train"],
        "validation": split["test"]
    })

# Load image processor
image_processor = AutoImageProcessor.from_pretrained("timm/convnext_base.fb_in22k_ft_in1k")
label_list = dataset["train"].features["label"].names
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

# Transform function
def transform(example):
    image = example["image"].convert("RGB")
    processed = image_processor(image, return_tensors="pt")
    example["pixel_values"] = processed["pixel_values"][0]
    return example

# Apply transformation
dataset = dataset.map(transform, remove_columns=["image"])

# Set format for PyTorch
dataset.set_format(type="torch", columns=["pixel_values", "label"])

# Load model
model = AutoModelForImageClassification.from_pretrained(
    "timm/convnext_base.fb_in22k_ft_in1k",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# Accuracy metric
accuracy = evaluate.load("accuracy")

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return accuracy.compute(predictions=preds, references=p.label_ids)

# Training arguments
training_args = TrainingArguments(
    output_dir="./plant-disease-model",
    logging_dir="./logs",
    per_device_train_batch_size=16,  # increase batch size
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,  # keep accumulation
    learning_rate=5e-5,
    warmup_ratio=0.1,
    max_grad_norm=1.0,
    weight_decay=0.02,
    num_train_epochs=20,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    logging_steps=10,
    fp16=True,
    bf16=torch.cuda.is_bf16_supported(),
    seed=42,
    report_to="tensorboard",
    optim="adamw_torch",
    lr_scheduler_type="cosine",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Save model
trainer.save_model("./plant-disease-model")

# Predictions
predictions = trainer.predict(dataset["validation"])
y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_list)
disp.plot(xticks_rotation=90)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")

# Accuracy plot
eval_acc = [log for log in trainer.state.log_history if "eval_accuracy" in log]
epochs = [x["epoch"] for x in eval_acc]
accs = [x["eval_accuracy"] for x in eval_acc]

plt.figure()
plt.plot(epochs, accs, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_plot.png")

print("✅ Training finished. Run: `tensorboard --logdir=./logs` to view logs.")
