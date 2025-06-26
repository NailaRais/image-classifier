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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import evaluate

# Load dataset (replace with your dataset if different)
dataset = load_dataset("Hemg/new-plant-diseases-dataset")

# Split dataset
if "validation" not in dataset:
    split = dataset["train"].train_test_split(test_size=0.2, seed=42)
    dataset = DatasetDict({
        "train": split["train"],
        "validation": split["test"]
    })

# Load image processor and model
model_ckpt = "timm/convnextv2_large.fcmae_ft_in22k_in1k"
image_processor = AutoImageProcessor.from_pretrained(model_ckpt)

label_list = dataset["train"].features["label"].names
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

# Preprocess images
def transform(example):
    image = example["image"].convert("RGB")
    processed = image_processor(image, return_tensors="pt")
    example["pixel_values"] = processed["pixel_values"][0]
    return example

dataset = dataset.map(transform, remove_columns=["image"])
dataset.set_format(type="torch", columns=["pixel_values", "label"])

# Load model
model = AutoModelForImageClassification.from_pretrained(
    model_ckpt,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# Metrics
accuracy = evaluate.load("accuracy")
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return accuracy.compute(predictions=preds, references=p.label_ids)

# Training arguments
training_args = TrainingArguments(
    output_dir="./plant-disease-model",
    logging_dir="./logs",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=20,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    learning_rate=5e-5,
    weight_decay=0.02,
    warmup_ratio=0.1,
    max_grad_norm=1.0,
    fp16=True,
    bf16=torch.cuda.is_bf16_supported(),
    load_best_model_at_end=True,
    seed=42,
    report_to="tensorboard",
    optim="adamw_torch",
    lr_scheduler_type="cosine"
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

# Evaluate and visualize
predictions = trainer.predict(dataset["validation"])
y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)

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

print("\u2705 Training complete! Run: `tensorboard --logdir=./logs` to view logs.")
