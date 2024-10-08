contrastive-dimension-estimation
======

This is the anonymous repository for the "Contrastive dimension reduction: when and how?" submission to NeurIPS 2024 and contains methods and experiments for contrastive dimension estimation. The purpose of the CDE function is to find the contrastive dimension between a foreground dataset and a background dataset. In other words, the methods quantify the information unique to the foreground group with respect to the background group. 



Dependency
------------
- Python 3.9.7
- anndata                  ( >= 0.10.7 )
- matplotlib               ( >= 3.8.4 )
- numpy                    ( >= 1.26.4)
- pandas                   ( >= 2.2.2 )
- Pillow                   ( >= 8.4.0 ) 
- scikit-dimension         ( >= 0.3.4 ) 
- scikit-learn             ( >= 0.24.2 )
- scipy                    ( >= 1.13.0 ) 
- seaborn                  ( >= 0.13.2 ) 


Note
-----------
In the code,  d1 refers to d_x and d2 refers to d_y, and the CD refers to d_xy in the paper. 

For more details about the CC-BY-4.0 license, please refer to LICENSE file in Github, or the Creative Commons website (https://creativecommons.org/2014/01/07/plaintext-versions-of-creative-commons-4-0-licenses/)