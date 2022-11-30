import hydra
import os
from omegaconf import DictConfig
from src.quantization import ErrorAwareQmnAnalyzer
from src.QExplorer import RandomExplorer, GoodSampleExplorer
from src.models import Model
from src.regressor import RFRegressor

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg:DictConfig)->None:
    os.environ["CUDA_VISIBLE_DEVICES"]=str(cfg.gpu)
    
    '''
    TODO
        - optimizers
        - goodSampleExplorer
        - move main in a dedicate algo
        - documentation
    '''
    m : Model
    m = hydra.utils.instantiate(cfg.model)['model'] 

    data = hydra.utils.instantiate(cfg.dataset)['dataset']

    qmn : ErrorAwareQmnAnalyzer
    qmn = hydra.utils.instantiate(cfg.qmnAnalyzer)['qmnAnalyzer']
    qmn.compute_stats(cfg.bitSpace)
    qmn.compute_accuracies(data,cfg.bitSpace)

    eRandom : RandomExplorer
    eRandom = hydra.utils.instantiate(cfg.explorer)['explorer']
    random_dataset = eRandom.explore(qmn.stats)

    eGood : GoodSampleExplorer
    eGood = hydra.utils.instantiate(cfg.explorer)['explorer']
    good_dataset = eGood.explore(qmn.stats,qmn.accuracies)

    total_dataset = random_dataset+good_dataset

    regr : RFRegressor
    regr = hydra.utils.instantiate(cfg.regressor)['regressor']
    regr.train(total_dataset)

if __name__=='__main__':
    main()