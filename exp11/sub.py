#BCELoss　Freez Paramater
import os
import datetime
import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
import torch.optim as optim
from omegaconf import DictConfig
import wandb
from wandb import plot
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel
from transformers import BertJapaneseTokenizer


# Dataset
class CreateDataset(Dataset):  # 文章のtokenize処理を行ってDataLoaderに渡す関数
    TEXT_COLUMN = "chunk"
    LABEL_COLUMN = "labels"
    FLAG_COLUMN = "flag"

    def __init__(self, data, tokenizer, max_token_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]  # iloc(data-frameの列の取得)/行数の取得
        text = data_row[self.TEXT_COLUMN]  # 行数分のtextを取得
        labels = data_row[self.LABEL_COLUMN]
        flags = data_row[self.FLAG_COLUMN]

        labels = labels.replace("[", "").replace("]", "")  # "[", "]" を削除

        # カンマで分割して各要素を取得し、int に変換して新しいリストに追加
        labels = [int(x.strip()) for x in labels.split(",")]

        encoding = self.tokenizer.encode_plus(  # encodingの詳細設定
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",  # pytorchに入力するように調整
        )

        return dict(
            text=text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.tensor(labels),
            #flags=torch.tensor(flags),
        )


# DataModule
class CreateDataModule(pl.LightningDataModule):
    """
    DataFrameからモデリング時に使用するDataModuleを作成
    バッチの分割、ランダムな抽出などを行うことでデータ処理とモデル訓練の分離を容易にする
    """

    def __init__(
        self,
        train_df,
        valid_df,
        test_df,
        batch_size,
        max_token_len,
        pretrained_model="cl-tohoku/bert-base-japanese-char-whole-word-masking",
    ):
        super().__init__()  # lightningdatamoduleのメソッドの継承
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_model)

    # train,val,testデータのsplit
    def setup(self, stage=None):
        self.train_dataset = CreateDataset(
            self.train_df, self.tokenizer, self.max_token_len
        )
        self.vaild_dataset = CreateDataset(
            self.valid_df, self.tokenizer, self.max_token_len
        )
        self.test_dataset = CreateDataset(
            self.test_df, self.tokenizer, self.max_token_len
        )

    #
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.vaild_dataset,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            pin_memory=True,
        )


# Model
class MaltiLabelClassifierModel(pl.LightningModule):
    THRESHOLD = 0.5  # 閾値

    def __init__(
        self,
        hidden_size,
        hidden_size2,
        num_classes,
        n_epochs=None,
        pretrained_model="cl-tohoku/bert-base-japanese-char-whole-word-masking",
    ):
        super(MaltiLabelClassifierModel, self).__init__()
        self.num_classes = num_classes
        self.train_step_outputs_preds = []
        self.train_step_outputs_labels = []
        self.validation_step_outputs_preds = []
        self.validation_step_outputs_labels = []
        self.test_step_outputs_preds = []
        self.test_step_outputs_labels = []

        # モデルの構造
        self.bert = BertModel.from_pretrained(pretrained_model, return_dict=True)

        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.bert.config.hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size2),
                nn.ReLU(),
                nn.Linear(hidden_size2, 1)
            ) for _ in range(num_classes)
        ])
        self.sigmoid = nn.Sigmoid()
        self.n_epochs = n_epochs
        self.criterion = nn.BCELoss()

        self.metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.Accuracy(
                    task="multilabel",
                    num_labels=16,
                    threshold=self.THRESHOLD,
                    average="macro",
                ),
                torchmetrics.Precision(
                    task="multilabel",
                    num_labels=16,
                    threshold=self.THRESHOLD,
                    average="macro",
                ),
                torchmetrics.Recall(
                    task="multilabel",
                    num_labels=16,
                    threshold=self.THRESHOLD,
                    average="macro",
                ),
                torchmetrics.F1Score(
                    task="multilabel",
                    num_labels=16,
                    threshold=self.THRESHOLD,
                    average="macro",
                ),
                torchmetrics.MatthewsCorrCoef(
                    task="multilabel", num_labels=16, threshold=self.THRESHOLD
                ),
            ]
        )

        self.metrics_per_label_accuracy = torchmetrics.MetricCollection(
            {
                f"accuracy_label_{i}": torchmetrics.Accuracy(
                    task="binary", num_labels=1, threshold=self.THRESHOLD
                )
                for i in range(num_classes)
            },
        )

        self.metrics_per_label_precision = torchmetrics.MetricCollection(
            {
                f"precision_label_{i}": torchmetrics.Precision(
                    task="binary", num_labels=1, threshold=self.THRESHOLD
                )
                for i in range(num_classes)
            },
        )

        self.metrics_per_label_recall = torchmetrics.MetricCollection(
            {
                f"recall_label_{i}": torchmetrics.Recall(
                    task="binary", num_labels=1, threshold=self.THRESHOLD
                )
                for i in range(num_classes)
            },
        )

        self.metrics_per_label_f1score = torchmetrics.MetricCollection(
            {
                f"f1score_label_{i}": torchmetrics.F1Score(
                    task="binary", num_labels=1, threshold=self.THRESHOLD
                )
                for i in range(num_classes)
            },
        )

        # BertLayerモジュールの最後を勾配計算ありに変更
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True

    # 順伝搬
    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)

        logits = [classifier(output.pooler_output) for classifier in self.classifiers]
        combine_outputs = torch.cat(logits, dim=1)  # 各クラスのバイナリ出力を結合

        preds = self.sigmoid(combine_outputs)
        loss = 0
        if labels is not None:
            loss = self.criterion(preds, labels.float())  # labelsをfloat型に変更する
        return loss, preds

    # trainのミニバッチに対して行う処理
    def training_step(self, batch, batch_idx):
        loss, preds = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            flags=batch["flag"]
        )

        self.train_step_outputs_preds.append(preds)
        self.train_step_outputs_labels.append(batch["labels"])
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        return {"loss": loss, "batch_preds": preds, "batch_labels": batch["labels"]}

    # validation、testでもtrain_stepと同じ処理を行う
    def validation_step(self, batch, batch_idx):
        loss, preds = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.validation_step_outputs_preds.append(preds)
        self.validation_step_outputs_labels.append(batch["labels"])
        # self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return {"loss": loss, "batch_preds": preds, "batch_labels": batch["labels"]}

    def test_step(self, batch, batch_idx):
        loss, preds = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.test_step_outputs_preds.append(preds)
        self.test_step_outputs_labels.append(batch["labels"])
        # self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        return {"loss": loss, "batch_preds": preds, "batch_labels": batch["labels"]}

    # epoch終了時にtrainのlossを記録
    def on_train_epoch_end(self, mode="train"):
        epoch_preds = torch.cat(self.train_step_outputs_preds)
        epoch_preds = epoch_preds.squeeze()
        epoch_labels = torch.cat(self.train_step_outputs_labels)
        epoch_labels = epoch_labels.squeeze()
        epoch_loss = self.criterion(epoch_preds, epoch_labels.float())
        self.log(f"{mode}_loss", epoch_loss, logger=True)
        class_names = [
            "あいづち",
            "感心",
            "評価",
            "繰り返し応答",
            "同意",
            "納得",
            "驚き",
            "言い換え",
            "意見",
            "考えている最中",
            "不同意",
            "補完",
            "あいさつ",
            "想起",
            "驚きといぶかり",
            "その他",
        ]

        metrics = self.metrics(epoch_preds, epoch_labels)
        for metric in metrics.keys():
            self.log(f"{mode}/{metric.lower()}", metrics[metric].item(), logger=True)

        for i in range(self.num_classes):
            label_preds = epoch_preds[:, i]  # i番目の要素のみを抽出
            label_labels = epoch_labels[:, i]
            metrics_per_label_accuracy = self.metrics_per_label_accuracy(
                label_preds, label_labels
            )
            metrics_per_label_precision = self.metrics_per_label_precision(
                label_preds, label_labels
            )
            metrics_per_label_recall = self.metrics_per_label_recall(
                label_preds, label_labels
            )
            metrics_per_label_f1score = self.metrics_per_label_f1score(
                label_preds, label_labels
            )
            self.log(
                f"{mode}/accuracy_label_{class_names[i]}",
                metrics_per_label_accuracy[f"accuracy_label_{i}"].item(),
                logger=True,
            )
            self.log(
                f"{mode}/presicion_label_{class_names[i]}",
                metrics_per_label_precision[f"precision_label_{i}"].item(),
                logger=True,
            )
            self.log(
                f"{mode}/recall_label_{class_names[i]}",
                metrics_per_label_recall[f"recall_label_{i}"].item(),
                logger=True,
            )
            self.log(
                f"{mode}/f1score_label_{class_names[i]}",
                metrics_per_label_f1score[f"f1score_label_{i}"].item(),
                logger=True,
            )

        self.train_step_outputs_preds.clear()  # free memory
        self.train_step_outputs_labels.clear()  # free memory

    # epoch終了時にvalidationのlossとaccuracyを記録
    def on_validation_epoch_end(
        self, mode="val"
    ):  # https://github.com/Lightning-AI/lightning/pull/16520
        # loss計算
        epoch_preds = torch.cat(self.validation_step_outputs_preds)
        epoch_preds = epoch_preds.squeeze()
        epoch_labels = torch.cat(self.validation_step_outputs_labels)
        epoch_labels = epoch_labels.squeeze()
        epoch_loss = self.criterion(epoch_preds, epoch_labels.float())
        self.log(f"{mode}_loss", epoch_loss, logger=True)

        class_names = [
            "あいづち",
            "感心",
            "評価",
            "繰り返し応答",
            "同意",
            "納得",
            "驚き",
            "言い換え",
            "意見",
            "考えている最中",
            "不同意",
            "補完",
            "あいさつ",
            "想起",
            "驚きといぶかり",
            "その他",
        ]

        metrics = self.metrics(epoch_preds, epoch_labels)
        for metric in metrics.keys():
            self.log(f"{mode}/{metric.lower()}", metrics[metric].item(), logger=True)

        for i in range(self.num_classes):
            label_preds = epoch_preds[:, i]  # i番目の要素のみを抽出
            label_labels = epoch_labels[:, i]
            metrics_per_label_accuracy = self.metrics_per_label_accuracy(
                label_preds, label_labels
            )
            metrics_per_label_precision = self.metrics_per_label_precision(
                label_preds, label_labels
            )
            metrics_per_label_recall = self.metrics_per_label_recall(
                label_preds, label_labels
            )
            metrics_per_label_f1score = self.metrics_per_label_f1score(
                label_preds, label_labels
            )
            self.log(
                f"{mode}/accuracy_label_{class_names[i]}",
                metrics_per_label_accuracy[f"accuracy_label_{i}"].item(),
                logger=True,
            )
            self.log(
                f"{mode}/presicion_label_{class_names[i]}",
                metrics_per_label_precision[f"precision_label_{i}"].item(),
                logger=True,
            )
            self.log(
                f"{mode}/recall_label_{class_names[i]}",
                metrics_per_label_recall[f"recall_label_{i}"].item(),
                logger=True,
            )
            self.log(
                f"{mode}/f1score_label_{class_names[i]}",
                metrics_per_label_f1score[f"f1score_label_{i}"].item(),
                logger=True,
            )

        self.validation_step_outputs_preds.clear()  # free memory
        self.validation_step_outputs_labels.clear()  # free memory

    # testデータのlossとaccuracyを算出
    def on_test_epoch_end(self, mode="test"):
        epoch_preds = torch.cat(self.test_step_outputs_preds)
        epoch_preds = epoch_preds.squeeze()
        epoch_labels = torch.cat(self.test_step_outputs_labels)
        epoch_labels = epoch_labels.squeeze()
        epoch_loss = self.criterion(epoch_preds, epoch_labels.float())
        self.log(f"{mode}_loss", epoch_loss, logger=True)

        class_names = [
            "あいづち",
            "感心",
            "評価",
            "繰り返し応答",
            "同意",
            "納得",
            "驚き",
            "言い換え",
            "意見",
            "考えている最中",
            "不同意",
            "補完",
            "あいさつ",
            "想起",
            "驚きといぶかり",
            "その他",
        ]

        metrics = self.metrics(epoch_preds, epoch_labels)
        for metric in metrics.keys():
            self.log(f"{mode}/{metric.lower()}", metrics[metric].item(), logger=True)

        for i in range(self.num_classes):
            label_preds = epoch_preds[:, i]  # i番目の要素のみを抽出
            label_labels = epoch_labels[:, i]
            metrics_per_label_accuracy = self.metrics_per_label_accuracy(
                label_preds, label_labels
            )
            metrics_per_label_precision = self.metrics_per_label_precision(
                label_preds, label_labels
            )
            metrics_per_label_recall = self.metrics_per_label_recall(
                label_preds, label_labels
            )
            metrics_per_label_f1score = self.metrics_per_label_f1score(
                label_preds, label_labels
            )
            self.log(
                f"{mode}/accuracy_label_{class_names[i]}",
                metrics_per_label_accuracy[f"accuracy_label_{i}"].item(),
                logger=True,
            )
            self.log(
                f"{mode}/presicion_label_{class_names[i]}",
                metrics_per_label_precision[f"precision_label_{i}"].item(),
                logger=True,
            )
            self.log(
                f"{mode}/recall_label_{class_names[i]}",
                metrics_per_label_recall[f"recall_label_{i}"].item(),
                logger=True,
            )
            self.log(
                f"{mode}/f1score_label_{class_names[i]}",
                metrics_per_label_f1score[f"f1score_label_{i}"].item(),
                logger=True,
            )

        tensor_list = torch.cat(self.test_step_outputs_preds)
        y_pred_flat = torch.reshape(tensor_list, [-1])  # 同様
        preds_binary = torch.where(y_pred_flat > self.THRESHOLD, 1, 0)
        preds = preds_binary.view(-1,16)
        dff = preds.cpu()
        df = pd.DataFrame(dff)
        #df.to_csv("table_exppp.csv", encoding="utf-8")
        #print('csv is done¥nß')

        self.test_step_outputs_preds.clear()
        self.test_step_outputs_labels.clear()  # free memory

    # optimizerの設定
    def configure_optimizers(self):
        # pretrainされているbert最終層のlrは小さめ、pretrainされていない分類層のlrは大きめに設定

        optimizer = optim.Adam(
            [
                {"params": self.bert.encoder.layer[-1].parameters(), "lr": 5e-5},
                {"params": self.hidden_layer1.parameters(), "lr": 1e-4},
                {"params": self.hidden_layer2.parameters(), "lr": 1e-4},
                #{"params": self.layer3.parameters(), "lr": 1e-4},
            ]
        )



        return [optimizer]


# モデルの保存と更新のための関数
def make_callbacks(min_delta, patience, checkpoint_path):
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="BCELoss_{epoch}",
        # save_top_k=1,  #save_best_only
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=min_delta, patience=patience, mode="min"
    )

    progress_bar = RichProgressBar()

    return [early_stop_callback, checkpoint_callback, progress_bar]


# NewModel
class NEWMaltiLabelClassifierModel(pl.LightningModule):
    THRESHOLD = 0.5  # 閾値

    def __init__(
        self,
        hidden_size,
        hidden_size2,
        num_classes,
        n_epochs=None,
        pretrained_model="cl-tohoku/bert-base-japanese-char-whole-word-masking",
    ):
        super(MaltiLabelClassifierModel, self).__init__()
        self.num_classes = num_classes
        self.train_step_outputs_preds = []
        self.train_step_outputs_labels = []
        self.validation_step_outputs_preds = []
        self.validation_step_outputs_labels = []
        self.test_step_outputs_preds = []
        self.test_step_outputs_labels = []

        # モデルの構造
        self.bert = BertModel.from_pretrained(pretrained_model, return_dict=True)

        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.bert.config.hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size2),
                nn.ReLU(),
                nn.Linear(hidden_size2, 1)
            ) for _ in range(num_classes)
        ])
        self.sigmoid = nn.Sigmoid()
        self.n_epochs = n_epochs
        self.criterion = nn.BCELoss()

        self.metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.Accuracy(
                    task="multilabel",
                    num_labels=16,
                    threshold=self.THRESHOLD,
                    average="macro",
                ),
                torchmetrics.Precision(
                    task="multilabel",
                    num_labels=16,
                    threshold=self.THRESHOLD,
                    average="macro",
                ),
                torchmetrics.Recall(
                    task="multilabel",
                    num_labels=16,
                    threshold=self.THRESHOLD,
                    average="macro",
                ),
                torchmetrics.F1Score(
                    task="multilabel",
                    num_labels=16,
                    threshold=self.THRESHOLD,
                    average="macro",
                ),
                torchmetrics.MatthewsCorrCoef(
                    task="multilabel", num_labels=16, threshold=self.THRESHOLD
                ),
            ]
        )

        self.metrics_per_label_accuracy = torchmetrics.MetricCollection(
            {
                f"accuracy_label_{i}": torchmetrics.Accuracy(
                    task="binary", num_labels=1, threshold=self.THRESHOLD
                )
                for i in range(num_classes)
            },
        )

        self.metrics_per_label_precision = torchmetrics.MetricCollection(
            {
                f"precision_label_{i}": torchmetrics.Precision(
                    task="binary", num_labels=1, threshold=self.THRESHOLD
                )
                for i in range(num_classes)
            },
        )

        self.metrics_per_label_recall = torchmetrics.MetricCollection(
            {
                f"recall_label_{i}": torchmetrics.Recall(
                    task="binary", num_labels=1, threshold=self.THRESHOLD
                )
                for i in range(num_classes)
            },
        )

        self.metrics_per_label_f1score = torchmetrics.MetricCollection(
            {
                f"f1score_label_{i}": torchmetrics.F1Score(
                    task="binary", num_labels=1, threshold=self.THRESHOLD
                )
                for i in range(num_classes)
            },
        )

        # BertLayerモジュールの最後を勾配計算ありに変更
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True

    # 順伝搬
    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)

        logits = [classifier(output.pooler_output) for classifier in self.classifiers]
        combine_outputs = torch.cat(logits, dim=1)  # 各クラスのバイナリ出力を結合

        preds = self.sigmoid(combine_outputs)
        loss = 0
        if labels is not None:
            loss = self.criterion(preds, labels.float())  # labelsをfloat型に変更する
        return loss, preds

    # trainのミニバッチに対して行う処理
    def training_step(self, batch, batch_idx):
        loss, preds = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            flags=batch["flag"]
        )

        # 特定の条件下でのパラメータの更新
        # 特定の条件下でのパラメータの更新
        for i,classifier in enumerate(self.classifiers):
            if flags[1][i] == 1:
                for param in classifier.parameters():
                    param.requires_grad = False

        self.train_step_outputs_preds.append(preds)
        self.train_step_outputs_labels.append(batch["labels"])
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        return {"loss": loss, "batch_preds": preds, "batch_labels": batch["labels"]}

    # validation、testでもtrain_stepと同じ処理を行う
    def validation_step(self, batch, batch_idx):
        loss, preds = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.validation_step_outputs_preds.append(preds)
        self.validation_step_outputs_labels.append(batch["labels"])
        # self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return {"loss": loss, "batch_preds": preds, "batch_labels": batch["labels"]}

    def test_step(self, batch, batch_idx):
        loss, preds = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.test_step_outputs_preds.append(preds)
        self.test_step_outputs_labels.append(batch["labels"])
        # self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        return {"loss": loss, "batch_preds": preds, "batch_labels": batch["labels"]}

    # epoch終了時にtrainのlossを記録
    def on_train_epoch_end(self, mode="train"):
        epoch_preds = torch.cat(self.train_step_outputs_preds)
        epoch_preds = epoch_preds.squeeze()
        epoch_labels = torch.cat(self.train_step_outputs_labels)
        epoch_labels = epoch_labels.squeeze()
        epoch_loss = self.criterion(epoch_preds, epoch_labels.float())
        self.log(f"{mode}_loss", epoch_loss, logger=True)
        class_names = [
            "あいづち",
            "感心",
            "評価",
            "繰り返し応答",
            "同意",
            "納得",
            "驚き",
            "言い換え",
            "意見",
            "考えている最中",
            "不同意",
            "補完",
            "あいさつ",
            "想起",
            "驚きといぶかり",
            "その他",
        ]

        metrics = self.metrics(epoch_preds, epoch_labels)
        for metric in metrics.keys():
            self.log(f"{mode}/{metric.lower()}", metrics[metric].item(), logger=True)

        for i in range(self.num_classes):
            label_preds = epoch_preds[:, i]  # i番目の要素のみを抽出
            label_labels = epoch_labels[:, i]
            metrics_per_label_accuracy = self.metrics_per_label_accuracy(
                label_preds, label_labels
            )
            metrics_per_label_precision = self.metrics_per_label_precision(
                label_preds, label_labels
            )
            metrics_per_label_recall = self.metrics_per_label_recall(
                label_preds, label_labels
            )
            metrics_per_label_f1score = self.metrics_per_label_f1score(
                label_preds, label_labels
            )
            self.log(
                f"{mode}/accuracy_label_{class_names[i]}",
                metrics_per_label_accuracy[f"accuracy_label_{i}"].item(),
                logger=True,
            )
            self.log(
                f"{mode}/presicion_label_{class_names[i]}",
                metrics_per_label_precision[f"precision_label_{i}"].item(),
                logger=True,
            )
            self.log(
                f"{mode}/recall_label_{class_names[i]}",
                metrics_per_label_recall[f"recall_label_{i}"].item(),
                logger=True,
            )
            self.log(
                f"{mode}/f1score_label_{class_names[i]}",
                metrics_per_label_f1score[f"f1score_label_{i}"].item(),
                logger=True,
            )

        self.train_step_outputs_preds.clear()  # free memory
        self.train_step_outputs_labels.clear()  # free memory

    # epoch終了時にvalidationのlossとaccuracyを記録
    def on_validation_epoch_end(
        self, mode="val"
    ):  # https://github.com/Lightning-AI/lightning/pull/16520
        # loss計算
        epoch_preds = torch.cat(self.validation_step_outputs_preds)
        epoch_preds = epoch_preds.squeeze()
        epoch_labels = torch.cat(self.validation_step_outputs_labels)
        epoch_labels = epoch_labels.squeeze()
        epoch_loss = self.criterion(epoch_preds, epoch_labels.float())
        self.log(f"{mode}_loss", epoch_loss, logger=True)

        class_names = [
            "あいづち",
            "感心",
            "評価",
            "繰り返し応答",
            "同意",
            "納得",
            "驚き",
            "言い換え",
            "意見",
            "考えている最中",
            "不同意",
            "補完",
            "あいさつ",
            "想起",
            "驚きといぶかり",
            "その他",
        ]

        metrics = self.metrics(epoch_preds, epoch_labels)
        for metric in metrics.keys():
            self.log(f"{mode}/{metric.lower()}", metrics[metric].item(), logger=True)

        for i in range(self.num_classes):
            label_preds = epoch_preds[:, i]  # i番目の要素のみを抽出
            label_labels = epoch_labels[:, i]
            metrics_per_label_accuracy = self.metrics_per_label_accuracy(
                label_preds, label_labels
            )
            metrics_per_label_precision = self.metrics_per_label_precision(
                label_preds, label_labels
            )
            metrics_per_label_recall = self.metrics_per_label_recall(
                label_preds, label_labels
            )
            metrics_per_label_f1score = self.metrics_per_label_f1score(
                label_preds, label_labels
            )
            self.log(
                f"{mode}/accuracy_label_{class_names[i]}",
                metrics_per_label_accuracy[f"accuracy_label_{i}"].item(),
                logger=True,
            )
            self.log(
                f"{mode}/presicion_label_{class_names[i]}",
                metrics_per_label_precision[f"precision_label_{i}"].item(),
                logger=True,
            )
            self.log(
                f"{mode}/recall_label_{class_names[i]}",
                metrics_per_label_recall[f"recall_label_{i}"].item(),
                logger=True,
            )
            self.log(
                f"{mode}/f1score_label_{class_names[i]}",
                metrics_per_label_f1score[f"f1score_label_{i}"].item(),
                logger=True,
            )

        self.validation_step_outputs_preds.clear()  # free memory
        self.validation_step_outputs_labels.clear()  # free memory

    # testデータのlossとaccuracyを算出
    def on_test_epoch_end(self, mode="test"):
        epoch_preds = torch.cat(self.test_step_outputs_preds)
        epoch_preds = epoch_preds.squeeze()
        epoch_labels = torch.cat(self.test_step_outputs_labels)
        epoch_labels = epoch_labels.squeeze()
        epoch_loss = self.criterion(epoch_preds, epoch_labels.float())
        self.log(f"{mode}_loss", epoch_loss, logger=True)

        class_names = [
            "あいづち",
            "感心",
            "評価",
            "繰り返し応答",
            "同意",
            "納得",
            "驚き",
            "言い換え",
            "意見",
            "考えている最中",
            "不同意",
            "補完",
            "あいさつ",
            "想起",
            "驚きといぶかり",
            "その他",
        ]

        metrics = self.metrics(epoch_preds, epoch_labels)
        for metric in metrics.keys():
            self.log(f"{mode}/{metric.lower()}", metrics[metric].item(), logger=True)

        for i in range(self.num_classes):
            label_preds = epoch_preds[:, i]  # i番目の要素のみを抽出
            label_labels = epoch_labels[:, i]
            metrics_per_label_accuracy = self.metrics_per_label_accuracy(
                label_preds, label_labels
            )
            metrics_per_label_precision = self.metrics_per_label_precision(
                label_preds, label_labels
            )
            metrics_per_label_recall = self.metrics_per_label_recall(
                label_preds, label_labels
            )
            metrics_per_label_f1score = self.metrics_per_label_f1score(
                label_preds, label_labels
            )
            self.log(
                f"{mode}/accuracy_label_{class_names[i]}",
                metrics_per_label_accuracy[f"accuracy_label_{i}"].item(),
                logger=True,
            )
            self.log(
                f"{mode}/presicion_label_{class_names[i]}",
                metrics_per_label_precision[f"precision_label_{i}"].item(),
                logger=True,
            )
            self.log(
                f"{mode}/recall_label_{class_names[i]}",
                metrics_per_label_recall[f"recall_label_{i}"].item(),
                logger=True,
            )
            self.log(
                f"{mode}/f1score_label_{class_names[i]}",
                metrics_per_label_f1score[f"f1score_label_{i}"].item(),
                logger=True,
            )

        tensor_list = torch.cat(self.test_step_outputs_preds)
        y_pred_flat = torch.reshape(tensor_list, [-1])  # 同様
        preds_binary = torch.where(y_pred_flat > self.THRESHOLD, 1, 0)
        preds = preds_binary.view(-1,16)
        dff = preds.cpu()
        df = pd.DataFrame(dff)
        #df.to_csv("table_exppp.csv", encoding="utf-8")
        #print('csv is done¥nß')

        self.test_step_outputs_preds.clear()
        self.test_step_outputs_labels.clear()  # free memory

    # optimizerの設定
    def configure_optimizers(self):
        # pretrainされているbert最終層のlrは小さめ、pretrainされていない分類層のlrは大きめに設定

        optimizer = optim.Adam(
            [
                {"params": self.bert.encoder.layer[-1].parameters(), "lr": 5e-5},
                {"params": self.hidden_layer1.parameters(), "lr": 1e-4},
                {"params": self.hidden_layer2.parameters(), "lr": 1e-4},
                #{"params": self.layer3.parameters(), "lr": 1e-4},
            ]
        )



        return [optimizer]


# モデルの保存と更新のための関数
def make_callbacks(min_delta, patience, checkpoint_path):
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="BCELoss_{epoch}",
        # save_top_k=1,  #save_best_only
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=min_delta, patience=patience, mode="min"
    )

    progress_bar = RichProgressBar()

    return [early_stop_callback, checkpoint_callback, progress_bar]


# Train Runner
@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    cwd = hydra.utils.get_original_cwd()
    current = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime(
        "%Y%m%d_%H%M%S"
    )

    # wandbの初期化
    wandb.init(
        project=cfg.wandb.project,
        name=("exp_" + str(cfg.wandb.exp_num)),
        tags=cfg.wandb.tags,
    )

    wandb_logger = WandbLogger(
        log_model=False,
    )
    # wandb_logger.watch(model, log="all")
    checkpoint_path = os.path.join(
        wandb_logger.experiment.dir, cfg.path.checkpoint_path
    )

    # dataModuleのインスタンス化
    train = pd.read_csv('/content/murata_labo_exp/data/chunk_prev_7_eda_flag1.csv')
    val,test = train_test_split(
        pd.read_csv(cfg.path.val_test_file_name),
        test_size=cfg.training.test_size,
        random_state=cfg.training.seed,
    )

    data_module = CreateDataModule(
        train,
        val,
        test,
        1,
        cfg.model.max_length,
    )
    data_module.setup()

    # callbackのインスタンス化
    call_backs = make_callbacks(
        cfg.callbacks.patience_min_delta, cfg.callbacks.patience, cfg.path.checkpoint_path
    )

    # modelのインスタンスの作成
    origin_model = MaltiLabelClassifierModel(
        hidden_size=cfg.model.hidden_size,
        hidden_size2=cfg.model.hidden_size2,
        num_classes=cfg.model.num_classes,
        n_epochs=cfg.training.n_epochs,
    )
    checkpoint_path = '/content/drive/MyDrive/murata_labo_exp/checkpoint/BCELoss_exp11.ckpt'
    original_model = MaltiLabelClassifierModel.load_from_checkpoint(checkpoint_path)

    new_model = NEWMaltiLabelClassifierModel(
        hidden_size=cfg.model.hidden_size,
        hidden_size2=cfg.model.hidden_size2,
        num_classes=cfg.model.num_classes,
        n_epochs=cfg.training.n_epochs,
    )
    new_model.load_state_dict(original_model.state_dict())

    # Trainerの設定
    trainer = pl.Trainer(
        max_epochs=cfg.training.n_epochs,
        devices="auto",
        # progress_bar_refresh_rate=30,
        callbacks=call_backs,
        logger=wandb_logger,
        fast_dev_run=False,
    )
    trainer.fit(new_model, data_module)
    trainer.test(new_model, data_module)

    wandb.finish()


if __name__ == "__main__":
    main()
