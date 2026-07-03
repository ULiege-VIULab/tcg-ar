"""
Orientation classifier (EfficientNet-B0, binary upright/180-degree).  Class-based
wrapper adapted from the cleaner ``identification_pipeline`` design.
"""

import torch
from torchvision import datasets, models
from torch.utils.data import DataLoader

from core.config import (ORIENTATION_TRAINING_SET_FOLDER_PATH, ORIENTATION_VALIDATION_SET_FOLDER_PATH,
                         ORIENTATION_TEST_SET_FOLDER_PATH, ORIENTATION_IMAGE_SIZE, ORIENTATION_BATCH_SIZE,
                         ORIENTATION_NUM_WORKERS, ORIENTATION_LR, ORIENTATION_EPOCHS, NB_CLASS_LABEL,
                         ORIENTATION_MODEL_FOLDER_PATH, ORIENTATION_MODEL_SAVE_PATH, OUTPUT_FOLDER_PATH,
                         ORIENTATION_DEFAULT_ARCH, ORIENTATION_ARCH_WEIGHTS)
from core.transforms import get_train_transform, get_valid_transform, get_inference_transform
from core.training_utils import (train_classifier_one_epoch, evaluate_classifier, save_model, save_plots)
from core.databases import create_nested_folders


class OrientationModel:
    def __init__(self):
        self.model = None

    def load_orientation_datasets(self):
        dataset_train = datasets.ImageFolder(ORIENTATION_TRAINING_SET_FOLDER_PATH, transform=get_train_transform(ORIENTATION_IMAGE_SIZE))
        dataset_valid = datasets.ImageFolder(ORIENTATION_VALIDATION_SET_FOLDER_PATH, transform=get_valid_transform(ORIENTATION_IMAGE_SIZE))
        # ImageFolder yields PIL images, so the test set uses the PIL-based valid
        # transform (get_inference_transform begins with ToPILImage, for raw ndarrays).
        dataset_test = datasets.ImageFolder(ORIENTATION_TEST_SET_FOLDER_PATH, transform=get_valid_transform(ORIENTATION_IMAGE_SIZE))
        return dataset_train, dataset_valid, dataset_test, dataset_train.classes

    def get_data_loaders(self, dataset_train, dataset_valid, dataset_test):
        train_loader = DataLoader(dataset_train, batch_size=ORIENTATION_BATCH_SIZE, shuffle=True, num_workers=ORIENTATION_NUM_WORKERS)
        valid_loader = DataLoader(dataset_valid, batch_size=ORIENTATION_BATCH_SIZE, shuffle=False, num_workers=ORIENTATION_NUM_WORKERS)
        test_loader = DataLoader(dataset_test, batch_size=ORIENTATION_BATCH_SIZE, shuffle=False, num_workers=ORIENTATION_NUM_WORKERS)
        return train_loader, valid_loader, test_loader

    def build_model(self, output_classes, device, arch=ORIENTATION_DEFAULT_ARCH):
        """Build a 2-way orientation classifier on the requested backbone. Only the
        architecture (and its final head) changes; everything else is shared so the
        comparison in the experiments isolates the backbone."""
        if arch == "efficientnet_b0":
            model = models.efficientnet_b0(weights='DEFAULT')
            model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=output_classes)
        elif arch == "resnet18":
            model = models.resnet18(weights='DEFAULT')
            model.fc = torch.nn.Linear(model.fc.in_features, output_classes)
        elif arch == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(weights='DEFAULT')
            model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, output_classes)
        elif arch == "shufflenet_v2_x1_0":
            model = models.shufflenet_v2_x1_0(weights='DEFAULT')
            model.fc = torch.nn.Linear(model.fc.in_features, output_classes)
        else:
            raise ValueError(f"Unknown orientation architecture: {arch}")
        for params in model.parameters():
            params.requires_grad = True
        model.to(device)
        self.model = model
        return model

    def train(self, work_dir=OUTPUT_FOLDER_PATH, arch=ORIENTATION_DEFAULT_ARCH, save_path=None):
        create_nested_folders(ORIENTATION_MODEL_FOLDER_PATH)
        if save_path is None:
            save_path = ORIENTATION_ARCH_WEIGHTS.get(arch, ORIENTATION_MODEL_SAVE_PATH)
        dataset_train, dataset_valid, dataset_test, dataset_classes = self.load_orientation_datasets()
        print(f"[INFO]: arch={arch}, training images: {len(dataset_train)}, validation: {len(dataset_valid)}, test: {len(dataset_test)}")
        print(f"[INFO]: Class names: {dataset_classes}")
        train_loader, valid_loader, test_loader = self.get_data_loaders(dataset_train, dataset_valid, dataset_test)

        device = ('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.build_model(NB_CLASS_LABEL, device, arch=arch)
        optimizer = torch.optim.Adam(model.parameters(), lr=ORIENTATION_LR)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5)
        criterion = torch.nn.CrossEntropyLoss()

        train_loss, valid_loss, train_acc, valid_acc = [], [], [], []
        for epoch in range(ORIENTATION_EPOCHS):
            print(f"[INFO]: Epoch {epoch+1} of {ORIENTATION_EPOCHS}")
            tl, ta = train_classifier_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
            vl, va = evaluate_classifier(model, valid_loader, criterion, device)
            train_loss.append(tl); valid_loss.append(vl); train_acc.append(ta); valid_acc.append(va)
            print(f"Training loss: {tl:.3f}, acc: {ta:.3f} | Validation loss: {vl:.3f}, acc: {va:.3f}")
            # Save a checkpoint after every epoch so progress can be followed and the
            # latest weights are always available (matches mmrotate's per-epoch saving).
            save_model(epoch + 1, model, optimizer, criterion, save_path)
            print(f"[INFO]: saved checkpoint after epoch {epoch+1} -> {save_path}")
            print('-' * 50)

        save_plots(train_acc, valid_acc, train_loss, valid_loss, work_dir, "orientation_" + arch)
        self.model = model
        print('TRAINING COMPLETE')

    def evaluate(self, arch=ORIENTATION_DEFAULT_ARCH):
        """Accuracy on the held-out synthetic orientation test set."""
        device = ('cuda' if torch.cuda.is_available() else 'cpu')
        if self.model is None:
            self.load_orientation_model(device, arch=arch)
        _, _, dataset_test, _ = self.load_orientation_datasets()
        test_loader = DataLoader(dataset_test, batch_size=ORIENTATION_BATCH_SIZE, shuffle=False, num_workers=ORIENTATION_NUM_WORKERS)
        criterion = torch.nn.CrossEntropyLoss()
        loss, acc = evaluate_classifier(self.model, test_loader, criterion, device)
        print(f"[{arch}] Orientation test loss: {loss:.3f}, accuracy: {acc:.3f}")
        return loss, acc

    def load_orientation_model(self, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                               arch=ORIENTATION_DEFAULT_ARCH, weight_path=None):
        if weight_path is None:
            weight_path = ORIENTATION_ARCH_WEIGHTS.get(arch, ORIENTATION_MODEL_SAVE_PATH)
        self.model = self.build_model(NB_CLASS_LABEL, device, arch=arch)
        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        return self.model
