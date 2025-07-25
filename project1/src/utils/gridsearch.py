from utils.dataloader import MameDataset
from utils.train import train_model, get_basic_train_cfg
from utils.modelcache import save_results


def grid_search(models, batches, dropouts, epochs, lrs, run):
  for model_type in models:
    for batch in batches:
      for dropout in dropouts:
        for n_epochs in epochs:
          for lr in lrs:
            # Get model instance
            model = model_type(num_classes=len(MameDataset.get_instance().classes),
                               dropout_ratio=dropout)
            name = f"{model_type.__name__}_dp{dropout}_e{n_epochs}_lr{lr}_run{run}"
            print(f"\n\n\n #### RUNNING: {name} ####")
            train_cfg = get_basic_train_cfg()
            train_cfg.batch = batch
            train_cfg.epochs = n_epochs
            train_cfg.lr = lr
            history = train_model(model, train_cfg=train_cfg, ptable_each=5)
            save_results(model, train_cfg, history, name)