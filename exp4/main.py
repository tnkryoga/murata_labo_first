#BCELoss not good model
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
from torchviz import make_dot
from pytorch_lightning.loggers import TensorBoardLogger



seed=42
torch.manual_seed(seed)

# Dataset
class CreateDataset(Dataset):  # 文章のtokenize処理を行ってDataLoaderに渡す関数
    TEXT_COLUMN = "chunk"
    LABEL_COLUMN = "labels"

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
        self.test_step_outputs_texts = []

        # モデルの構造
        self.bert = BertModel.from_pretrained(pretrained_model, return_dict=True)
        self.classifiers = nn.ModuleList(
            [
                nn.Linear(self.bert.config.hidden_size, hidden_size)
                for _ in range(num_classes)
            ]
        )  # 入力BERT層、出力hidden_sizeの全結合層/二値分類器をクラス数分並べる
        self.hidden_layer1 = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size2) for _ in range(num_classes)]
        )  # classifierの隠れ層の追加
        self.hidden_layer2 = nn.ModuleList(
            [nn.Linear(hidden_size2, 1) for _ in range(num_classes)]
        )  # classifierの隠れ層の追加
        self.sigmoid = nn.Sigmoid()
        self.n_epochs = n_epochs
        self.criterion = nn.BCELoss()
        
        self.print_initial_weights()

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

    def print_initial_weights(self):
        # classifiersの重みの初期値を出力
        # for i, classifier in enumerate(self.classifiers):
        #     print(f"Classifier {i+1} initial weights:")
        #     for name, param in classifier.named_parameters():
        #         if 'weight' in name:
        #             print(name, param.data)

        # # hidden_layer1の重みの初期値を出力
        # for i, layer in enumerate(self.hidden_layer1):
        #     print(f"Hidden Layer 1 - {i+1} initial weights:")
        #     for name, param in layer.named_parameters():
        #         if 'weight' in name:
        #             print(name, param.data)

        # # hidden_layer2の重みの初期値を出力
        # for i, layer in enumerate(self.hidden_layer2):
        #     print(f"Hidden Layer 2 - {i+1} initial weights:")
        #     for name, param in layer.named_parameters():
        #         if 'weight' in name:
        #             print(name, param.data)
        pass

    # 順伝搬
    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        hidden_outputs = []
        for classifier, hidden_layer1, hidden_layer2 in zip(
            self.classifiers, self.hidden_layer1, self.hidden_layer2
        ):
            binary_output = torch.relu(classifier(output.pooler_output))
            hidden_output1 = torch.relu(hidden_layer1(binary_output))
            hidden_output2 = hidden_layer2(hidden_output1)
            hidden_outputs.append(hidden_output2)

        combine_outputs = torch.cat(hidden_outputs, dim=1)  # 各クラスのバイナリ出力を結合
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
        )

        #print(batch["labels"])

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
        self.test_step_outputs_texts.append(batch["text"])
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

        # # 予測結果と正解ラベルをデータフレームに変換し、CSVに出力
        # texts = self.test_step_outputs_texts  # テキストはそのままリストとして使用
        # trans_text = [item for subtext in texts for item in subtext]
        # trans_pred = [item for subpred in self.test_step_outputs_preds for item in subpred]
        # binary_pred = [[1 if val > self.THRESHOLD else 0 for val in sublist] for sublist in trans_pred]
        # trans_label = [item for sublabels in self.test_step_outputs_labels for item in sublabels]
        # print(self.test_step_outputs_preds)
        # print(binary_pred)
        # print(len(trans_text))
        # print(len(binary_pred))
        # print(len(trans_label))
        # df = pd.DataFrame([
        #     trans_text,
        #     binary_pred,
        #     trans_label
        # ])
        # df = pd.DataFrame.transpose(df)
        # dff = df.rename(columns={0:'chunk',1:'pred_labels',2:'true_labels'})
        # dff.to_csv("predictions_5.csv", encoding="utf-8", index=False)
        # print('CSV is saved.')

        self.test_step_outputs_preds.clear()
        self.test_step_outputs_labels.clear()  # free memory
        self.test_step_outputs_texts.clear()  # 入力文章の保存をクリア


    # optimizerの設定
    def configure_optimizers(self):
        # pretrainされているbert最終層のlrは小さめ、pretrainされていない分類層のlrは大きめに設定
        optimizer = optim.Adam(
           self.parameters(),lr=1e-4
        )

        return [optimizer]


# モデルの保存と更新のための関数
def make_callbacks(min_delta, patience, checkpoint_path):
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="BCELoss_checker",
        save_top_k=1,  #save_best_only
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
    train = pd.read_csv(cfg.path.train_file_name)
    val,test = train_test_split(
        pd.read_csv(cfg.path.val_test_file_name),
        test_size=cfg.training.test_size,
        random_state=cfg.training.seed,
    )

    data_module = CreateDataModule(
        train,
        val,
        test,
        cfg.training.batch_size,
        cfg.model.max_length,
    )
    data_module.setup()

    # callbackのインスタンス化
    call_backs = make_callbacks(
        cfg.callbacks.patience_min_delta, cfg.callbacks.patience, cfg.path.checkpoint_path
    )

    # modelのインスタンスの作成
    model = MaltiLabelClassifierModel(
        hidden_size=cfg.model.hidden_size,
        hidden_size2=cfg.model.hidden_size2,
        num_classes=cfg.model.num_classes,
        n_epochs=cfg.training.n_epochs,
    )

    # dummy_input_ids = torch.randint(0, 1000, (1, cfg.model.max_length)).long()
    # dummy_attention_mask = torch.ones((1, cfg.model.max_length)).long()
    # dummy_labels = torch.zeros((1, cfg.model.num_classes)).long()

    # # 順伝搬の実行
    # model.eval()
    # with torch.no_grad():
    #     loss, preds = model(dummy_input_ids, dummy_attention_mask, dummy_labels)

    # # モデルのグラフを作成
    # make_dot(preds, params=dict(model.named_parameters())).render("model_structure", format="png")

    # main関数内でwandb_loggerの前にTensorBoardLoggerを追加
    tensorboard_logger = TensorBoardLogger("logs/", name="exp_" + str(cfg.wandb.exp_num))
    



   # モデル1の重みをロード
    state_dict_model1 = torch.load('/content/murata_labo_exp/checkpoint/BCELoss_exp12_good.ckpt')

    # モデル2のstate_dictを取得
    state_dict_model2 = model.state_dict()
    print(state_dict_model1)
    

    # モデル1の重みをモデル2の対応する層にコピー
    for name, param in state_dict_model1.items():
        if 'classifiers' in name:
            classifier_idx = int(name.split('.')[1])
            layer_idx = int(name.split('.')[2])
            if layer_idx == 0:
                # BERTの出力層からhidden_sizeへの全結合層
                state_dict_model2[f'classifiers.{classifier_idx}.weight'] = state_dict_model1[f'classifiers.{classifier_idx}.0.weight']
                state_dict_model2[f'classifiers.{classifier_idx}.bias'] = state_dict_model1[f'classifiers.{classifier_idx}.0.bias']
                if classifier_idx == 0:
                    print(param)
            elif layer_idx == 2:
                # hidden_sizeからhidden_size2への全結合層
                state_dict_model2[f'hidden_layer1.{classifier_idx}.weight'] = state_dict_model1[f'classifiers.{classifier_idx}.2.weight']
                state_dict_model2[f'hidden_layer1.{classifier_idx}.bias'] = state_dict_model1[f'classifiers.{classifier_idx}.2.bias']
            elif layer_idx == 4:
                # hidden_size2から出力層への全結合層
                state_dict_model2[f'hidden_layer2.{classifier_idx}.weight'] = state_dict_model1[f'classifiers.{classifier_idx}.4.weight']
                state_dict_model2[f'hidden_layer2.{classifier_idx}.bias'] = state_dict_model1[f'classifiers.{classifier_idx}.4.bias']
    
    model.load_state_dict(state_dict_model2)

    # Trainerの設定
    trainer = pl.Trainer(
        max_epochs=cfg.training.n_epochs,
        devices="auto",
        # progress_bar_refresh_rate=30,
        callbacks=call_backs,
        logger=[wandb_logger, tensorboard_logger],
        fast_dev_run=False,
    )
    #trainer.fit(model, data_module)
    trainer.test(model, data_module)

    wandb.finish()


if __name__ == "__main__":
    main()
