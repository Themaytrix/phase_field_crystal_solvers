#!/bin/bash                                                                                                                    

#SBATCH --job-name=pfc   # Job name                                                                               
#SBATCH --qos=jakar_medium_general                                                                                                              

#SBATCH -p medium                                                                                                         

#SBATCH -o output.txt                                                                                                          

#SBATCH -e error.txt                                                                                                           

#SBATCH --mail-type=all
#SBATCH --mail-user=robeng@miners.utep.edu    # Where to send mail                                                                    
                                                                            
module load gnu12/12.2.0 py3-numpy/1.19.5


python pfc.py
