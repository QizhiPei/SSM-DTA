# Preprocess from DeepAffinity

(This part is copied from [DeepAffinity](https://github.com/Shen-Lab/DeepAffinity/tree/master/data/script/split_data_script) and modified. Note that the Google Drive link provided by DeepAffinty is actually a sdf file, not a tsv file, which may need further modification)

1. Download dataset from bindingDB website in tsv format:
   https://drive.google.com/open?id=1_uZVTBVPeeF64joitPU8uDZnfIGS1me0
2. Unzip dataset to the same directory with script and run:
  python firstStep.py
3. Then run 
  python uniquePnC.py
  it will generate unique CID (uniqueCID) and unique protein (uniqueProtein) files.
4. Let's deal with compound data first. 
   move the uniqueCID file to pubchem directory:
   mv uniqueCID pubchem
   cd pubchem
   Based on the unique CID file, you need to get the sdf data from https://pubchem.ncbi.nlm.nih.gov/pc_fetch/pc_fetch.cgi
   This webserver has limitation of download number per request. You may need to split the unique CID file into several files.
   You can run:
   python splitCID.py 
   and it will help to split file with each one containing 200k CIDs.
   After downloading sdf file from webserver, please rename those files as "CID1.sdf", "CID2.sdf" and ect. Then run:
   python getCID_Feature.py
   which will integrate those files and convert features of protein from base64 format to binary number. In this step, please notice that if the number of split CID files is not 3, you need to modify the 30th line of "getCID_Feature.py" with the files number you get. For example, if you split into 4 CID files, it should be modified as "fileNum = 4".
5. Move the result file "CID_Smi_Feature" to the last directory. Run:
  mv CID_Smi_Feature ..
  cd ..
  python toCanSMILE.py
  and it helps to convert SMILE sequences to canonical SMILE format.
6. Split data. Copy the result file of each steps from corresponding folders:
  Compound feature dictionary: "CID_Smi_Feature" from folder "pubchem"
  Copy this file to the root directory and run:
  python split.py BindingDB_All_firststep_noMulti_can.tsv
  "BindingDB_All_firststep_noMulti_can.tsv" is the result of step 5
  Then you will get split data based on different measurements and classes.

