#INPUT=my_train1
INPUT=test1

# find control channels for each well
for((c=0;c<=30;c++)); do 
    for f in $(awk -v FS="_" '{print $1"_"$2}' $INPUT.set); do 
          if [ "$oldF" != $f ]; then 
                 A=$(grep $f controls/${c}_controls.csv | head -n 1); 
          fi; 
          echo $A; 
          oldF=$f; 
     done | cut -f 4 -d, > tmp/a_$c; 
     echo $c; 
done

cp $INPUT.set aaa; for((c=0;c<=30;c++)); do paste -d, aaa tmp/a_${c} > bbb; mv bbb aaa; done
mv aaa ${INPUT}_controls.set

# calculate cosines of each image and 31 corresponding controls into folder cc
# takes about 5-7 minutes
./calculate_feature_cosines.sh ${INPUT}_controls.set features_${INPUT}_1.csv features_${INPUT}_2.csv

paste cc/* -d, | sed -e 's/ /,/g' > cc.1
awk -v FS="," '{for(i=1;i<=NF/2;i++) printf("%f,", ($(2*i)+$(2*i-1))/2 ); print "";}' cc.1 > cc.3
(for((i=1;i<=31;i++)); do echo -n "C${i}_1,C${i}_2,"; done; echo; cat cc.3) | sed -e 's/,$//g' > cc.4

paste $INPUT.set cc.4 -d, > ${INPUT}B.set
