#!/usr/bin/env bash

cd $HOME/Research/Projects/FinalBodyMTS/remote/
rsync -avzhe 'ssh -J ashisig@ashisig_4XA100' --exclude-from 'ignore_files.txt' $HOME/Research/Projects/FinalBodyMTS/  ashisig@ashisig_4XA100:/home/ashisig/Research/Projects/FinalBodyMTS/

# rsync -avzhe 'ssh -J ashisig@ashisig_4XA100' --exclude-from 'ignore_files.txt' $HOME/Research/Data/  ashisig@ashisig_4XA100:/home/ashisig/Research/Data/



#rsync -avzhe 'ssh -J 19205522@resit-ssh.ucd.ie' /Users/ashishsingh/Results/Datasets/HPE/TrainTestData_70_30  19205522@login.ucd.ie:/home/people/19205522/scratch/Results/Datasets/HPE/

#rsync -avzhe 'ssh -J 19205522@resit-ssh.ucd.ie' --exclude-from 'ignore_files.txt' /Users/ashishsingh/PycharmProjects/sonic_testing/  19205522@login.ucd.ie:/home/people/19205522/Research/Codes/sonic_testing/


#rsync -avzhe 'ssh -J 19205522@resit-ssh.ucd.ie' /Users/ashishsingh/Downloads/temp/  19205522@login.ucd.ie:/home/people/19205522/scratch/temp/