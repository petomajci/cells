INPUT=$1   # my_train1_controls.set
FEATURE_FILE1=$2  # features_train1_1.csv
FEATURE_FILE2=$3  # features_train1_2.csv

for((x=6;x<=36;x++)); do   #36
  oldF="empty" 
  for f in $(awk -v x=$x -v FS="," '{print $2"_"$3"_"$x}' $INPUT | tail -n +2); do
     if [ ${f: -1} != "_" ]; then 
       if [ "$oldF" != $f ]; then 
          A=$(grep -n $f all_controls.csv | cut -d: -f 1);
          #echo f=$f   a=$A
	  B=$((A-1))
          F1=$(head -n $B features_controls_1.csv | tail -n 1)
          F2=$(head -n $B features_controls_2.csv | tail -n 1)
       fi; 
       echo $F1,$F2; 
       oldF=$f;
     else
       cat zero.row
       #echo f=$f  NO GREP
     fi 
  done > cc1


  paste -d, $FEATURE_FILE1 $FEATURE_FILE2 cc1 > cc2

  awk -v FS="," -f calculate_cosines.awk cc2 > cc/cc.$x
  date
  echo $x
done
