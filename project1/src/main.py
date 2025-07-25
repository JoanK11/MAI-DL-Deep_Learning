import sys
from utils.dataloader import MameDataset
from utils.gridsearch import grid_search
from utils.train import get_basic_train_cfg, train_model
from utils.modelcache import save_results
from models.zfnet import ZfNet
from models.googlenet import GoogLeNet
from models.customnet import CustomNet


if __name__ == "__main__":
  MameDataset.initialize_instance(testing=False)
  if "test" in sys.argv:
    model = CustomNet(len(MameDataset.get_instance().classes))
    train_cfg = get_basic_train_cfg()
    train_cfg.epochs = 10
    history = train_model(model, train_cfg, ptable_each=5)
    save_results(model, train_cfg, history, "test")
  else:
    for run in range(3):
      grid_search(models=[CustomNet, GoogLeNet],
                  batches=[512],
                  dropouts=[0.0, 0.5, 0.95],
                  epochs=[20, 80],
                  lrs=[0.001, 0.0001],
                  run=run+1)