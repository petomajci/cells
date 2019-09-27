vi ~/.bash_history 
awk -v FS="," -v OFS="," '{$5+=30; print $1,$2,$3,$4,$5}' all_controls.csv | less
vi my_train_30classesHUVEC.set 
cut -f 5 -d, my_train_30classesHUVEC.set | sort | uniq -c 
awk -v FS="," -v OFS="," '{$5+=30; print $1,$2,$3,$4,$5}' all_controls.csv > all_controls2.csv 
cat all_controls2.csv my_train_30classesHUVEC.set > my_train_30classesHUVEC2.set
vi my_train_30classesHUVEC2.set 
vi train_regular.py 
awk -v FS="," -v OFS="," '{$5+=30; print $1,$2,$3,$4,$5}' all_controls.csv | grep HUVEC > all_controls2.csv 
cat all_controls2.csv my_train_30classesHUVEC.set > my_train_30classesHUVEC2.set
wc -l my_train_30classesHUVEC2.set
vi my_train_30classesHUVEC2.set 
vi train_regular.py 
grep -n HUVEC all_controls.csv | less
tail -n +2000 all_controls.csv | awk -v FS="," -v OFS="," '{$5+=30; print $1,$2,$3,$4,$5}' | grep HUVEC > all_controls2.csv 
wc -l all_controls2.csv 
wc -l my_train_30classesHUVEC.set 
less all_controls2.csv
cat all_controls2.csv my_train_30classesHUVEC.set > my_train_30classesHUVEC2.set
vi my_train_30classesHUVEC2.set
vi train_regular.py 
vi ImagesDS.py 
vi my_train1.set 
ls -ltr
vi train_controls.csv 
grep negative train_controls.csv | wc -l
grep negative train_controls.csv | less
grep negative train_controls.csv > negative.controls
vi my_train1.set 
awk -v FS="_" '{print $1"_"$2}' my_train1.set | less
awk -v FS="_" '{print $1"_"$2}' my_train1.set | wc -l
for f in $(awk -v FS="_" '{print $1"_"$2}' my_train1.set | head); do echo $f; done
for f in $(awk -v FS="_" '{print $1"_"$2}' my_train1.set | head); do grep $f negative.controls; done
vi negative.controls 
grep negative train_controls.csv test_controls.csv > negative.controls
for f in $(awk -v FS="_" '{print $1"_"$2}' my_train1.set | head); do grep $f negative.controls; done
grep HUVEC-18_3 negative.controls
man grep
grep -h HUVEC-18_3 negative.controls
grep "HUVEC-18_3" negative.controls
man grep
vi ~/.bashrc 
grep "HUVEC-18_3" negative.controls 
grep -h "HUVEC-18_3" negative.controls 
grep --no-filename "HUVEC-18_3" negative.controls 
vi negative.controls 
grep -h negative train_controls.csv test_controls.csv > negative.controls
grep "HUVEC-18_3" negative.controls 
for f in $(awk -v FS="_" '{print $1"_"$2}' my_train1.set | head); do grep $f negative.controls; done
for f in $(awk -v FS="_" '{print $1"_"$2}' my_train1.set | head); do A=$(grep $f negative.controls); echo $A; done
for f in $(awk -v FS="_" '{print $1"_"$2}' my_train1.set | head); do if [ "$oldF" != $f ]; then A=$(grep $f negative.controls); fi; echo $A; done
for f in $(awk -v FS="_" '{print $1"_"$2}' my_train1.set | head); do if [ "$oldF" != $f ]; then A=$(grep $f negative.controls); fi; echo $A; oldF=$f; done
for f in $(awk -v FS="_" '{print $1"_"$2}' my_train1.set); do if [ "$oldF" != $f ]; then A=$(grep $f negative.controls); fi; echo $A; oldF=$f; done | wc -l
for f in $(awk -v FS="_" '{print $1"_"$2}' my_train1.set); do if [ "$oldF" != $f ]; then A=$(grep $f negative.controls); fi; echo $A; oldF=$f; done | less
for f in $(awk -v FS="_" '{print $1"_"$2}' my_train1.set); do if [ "$oldF" != $f ]; then A=$(grep $f negative.controls | head -n 1); fi; echo $A; oldF=$f; done | wc -l
for f in $(awk -v FS="_" '{print $1"_"$2}' my_train1.set); do if [ "$oldF" != $f ]; then A=$(grep $f negative.controls | head -n 1); fi; echo $A; oldF=$f; done > a1
for f in $(awk -v FS="_" '{print $1"_"$2}' my_train1.set); do if [ "$oldF" != $f ]; then A=$(grep $f negative.controls | tail -n 1); fi; echo $A; oldF=$f; done > a2
diff a1 a2 | wc -l
diff a1 a2 | less
vi a1
wc -l my_train1.set a1
paste my_train1.set a1 | less
man paste
paste -d, my_train1.set a1 | less
for f in $(awk -v FS="_" '{print $1"_"$2}' my_train1.set); do if [ "$oldF" != $f ]; then A=$(grep $f negative.controls | head -n 1); fi; echo $A; oldF=$f; done | cut -f 4 -d, > a1
paste my_train1.set a1 | less
paste -d, my_train1.set a1 | less
paste -d, my_train1.set a1 > my_train1N.set
vi my_train1N.set
vi ImagesDS.py 
cp ImagesDS.py ImagesDS_negative.py
vi ImagesDS_negative.py 
cp train_regular.py train_negative.py
vi train_negative.py 
cp DensNet.py DensNet_negative.py 
vi DensNet_negative.py 
vi train_negative.py 
wc -l *set
for f in $(awk -v FS="_" '{print $1"_"$2}' my_train_30classesHUVEC.set); do if [ "$oldF" != $f ]; then A=$(grep $f negative.controls | head -n 1); fi; echo $A; oldF=$f; done | cut -f 4 -d, > a1
wc -l a1
paste -d, my_train_30classesHUVEC.set a1 > my_train_30classesHUVEC_N.set
vi my_train_30classesHUVEC_N.set
vi train_negative.py 
vi ImagesDS_negative.py 
vi DensNet_negative.py 
vi my_train_30classesHUVEC_N.set
vi train_negative.py 
vi DensNet_negative.py 
vi my_train_30classesHUVEC_N.set
awk -v FS="_" '{print $1"_"$2}' my_train_30classesHUVEC.set | less
awk -v FS="_" '{print $1"_"$2}' my_train_30classesHUVEC.set | head
grep HUVEC-01_1 negative.controls 
grep HUVEC-01_2 negative.controls 
grep HUVEC-01_3 negative.controls 
grep HUVEC-01_4 negative.controls 
grep HUVEC-02_1 negative.controls 
grep HUVEC-02_2 negative.controls 
grep HUVEC-02_3 negative.controls 
vi a1
sort a1 | uniq -c
vi train_controls.csv 
grep negative train_controls.csv | less
grep negative train_controls.csv | cut -f 4 -d, | less
grep negative train_controls.csv | cut -f 4 -d, | sort | uniq -c
ls -ltr
vi my_train_30classesHUVEC_N.set
vi ImagesDS_negative.py 
vi DensNet_negative.py 
vi ImagesDS_negative.py 
vi train_negative.py 
vi ImagesDS_negative.py 
vi DensNet_negative.py 
vi train_negative.py 
vi DensNet_negative.py 
vi ImagesDS_negative.py 
vi DensNet_negative.py 
exit
ls
ls -ltr
python train_regular.py 2 final_model_1.bin all_controls.csv 30
python train_regular.py 1H final_model_1.bin my_train_30classesHUVEC.set 30
python train_regular.py 2H final_model_1H.bin my_train_30classesHUVEC.set 100
python train_regular.py 1H none my_train_30classesHUVEC.set 100
python train_regular.py 1S none my_train_30classesHUVEC.set 100
python train_regular.py 1S none my_train_30classesHUVEC2.set 100
python train_regular.py 1N none my_train_30classesHUVEC_N.set 100
python train_negative.py 1N none my_train_30classesHUVEC_N.set 100
exit
sudo mount -o discard,defaults /dev/sdb /data
cd /data/code/cells/
ls -ltr *py
cp train_regular.py train_small.py
vi train_small.py 
vi DensNet.py 
sudo mount -o discard,defaults /dev/sdb /data
cd /data/code/cells/
vi DensNet.py 
vi train_small.py 
vi DensNet.py 
vi train_small.py 
vi ImagesDS.py 
vi DensNet.py 
python train_small.py 1S none my_train_30classesHUVEC.set 100
cd /data/code/cells/
python train_small.py 1S none my_train_30classesHUVEC.set 100
sudo mount -o discard,defaults /dev/sdb /data
cd /data/code/cells/
vi DensNet.py 
grep BOTH *.py
vi train_small.py 
vi DensNet.py 
vi train_small.py 
vi DensNet.py 
ls -ltra /home/pmajek/.cache/torch/checkpoints/ 
ls -ltrh /home/pmajek/.cache/torch/checkpoints/ 
top
vi DensNet.py 
top
vi train_small.py 
grep num_workers *
vi train_regular.py 
vi train_negative.py 
top
vi train_small.py 
vi train_cosFace.py 
vi cosface.py
vi train_cosFace.py 
vi cosface.py
vi train_cosFace.py 
vi train_small.py 
vi train_cosFace.py 
vi train_small.py 
vi train_cosFace.py 
vi train_small.py 
vi train_cosFace.py 
vi train_small.py 
vi train_regular.py 
vi train_cosFace.py 
vi cosface.py
vi train_cosFace.py 
vi DensNet.py 
vi train_cosFace.py 
cp DensNet.py DensNet_forCosFace.py
vi train_cosFace.py 
vi DensNet_forCosFace.py 
vi train_cosFace.py 
vi train_regular.py 
vi train_cosFace.py 
ls *set
ls ../../input/
ls -ltr
vi all_controls2.csv
vi train_cosFace.py 
wc -l train_cosFace.py
wc -l all_controls.csv
wc -l train_cosFace.py
vi train_cosFace.py 
vi train_regular.py 
vi train_cosFace.py 
vi all_controls.csv 
vi cosface.py 
cp cosface.py cosface2D.py 
vi cosface2D.py 
vi cosface.py 
vi train_cosFace.py 
vi cosface.py 
vi train_cosFace.py 
ls -ltr
vi train_cosFace.py 
cd /data/code/cells/
python train_small.py 1S none my_train_30classesHUVEC.set 100
python train_cosFace.py 1S none my_train_30classesHUVEC.set 100
python train_cosFace.py 1S none all_controls.csv 100
python train_cosFace.py 1S none my_train_30classesHUVEC.set 100
python train_cosFace.py 1S none all_controls.csv 100
top
cd /data/code/cells/ls -ltr
cd /data/code/cells/
ls -ltr
vi train_cosFace.py 
sudo mount -o discard,defaults /dev/sdb /data
cd /data/code/
cd cells/
ls -ltr
vi train_cosFace.py 
mv best_centers_cosFace_loss\{groupCode\}.bin best_centers_cosFace_loss1S.bin 
mv best_centers_cosFace_acc\{groupCode\}.bin best_centers_cosFace_acc1S.bin 
sl -ltr
ls -ltr
screen -list
python train_cosFace.py 1S none all_controls.csv 100
cd /data/code/cells/
git status
git add *py
git status
vi .git/
vi .gitignore
git status
git add .gitignore 
ls *.1
ll *.1
ls -l *.1
git add sirnas.*
git status
git commit -a
git push
git pull
git status
git push
vi train_cosFace.py 
cd /data/code/cells/
ls -ltr
python train_cosFace.py 1S best_model_cosFace_loss1S.bin best_centers_cosFace_loss1S.bin  all_controls.csv 100
python train_cosFace.py 2D none none  all_controls.csv 100
python features_cosFace.py 2D best_model_cosFace_acc2D.bin best_centers_cosFace_acc2D.bin  all_controls.csv 100
vi features.csv 
exit
sudo mount -o discard,defaults /dev/sdb /data
cd /data/code/cells/
vi train_cosFace.py 
vi DensNet_forCosFace.py 
vi train_cosFace.py 
vi cosface2D.py 
vi train_cosFace.py 
top
vi train_cosFace.py 
vi cosface2D.py 
vi train_cosFace.py 
vi cosface2D.py 
top
ls -ltr
ls __pycache__/
ls -ltr __pycache__/
ls -ltr
vi train_cosFace.py 
cp train_cosFace.py features_cosFace.py
vi features_cosFace.py 
vi cosface2D.py 
vi features_cosFace.py 
vi train_cosFace.py 
vi features_cosFace.py 
ls -ltr
vi features_cosFace.py 
ls -ltr
vi preds.csv 
vi all_controls.csv 
vi features_cosFace.py 
exit
cd /data/code/cells/
vi features.csv 
python features_cosFace.py 2D best_model_cosFace_acc2D.bin best_centers_cosFace_acc2D.bin  all_controls.csv 100
vi features.csv 
python features_cosFace.py 2D best_model_cosFace_acc2D.bin best_centers_cosFace_acc2D.bin  all_controls.csv 100
vi features.csv 
python features_cosFace.py 2D best_model_cosFace_acc2D.bin best_centers_cosFace_acc2D.bin  all_controls.csv 100
vi features.csv 
wc -l features.csv 
python features_cosFace.py 2D best_model_cosFace_acc2D.bin best_centers_cosFace_acc2D.bin  all_controls.csv 100
wc -l features.csv 
vi features.csv 
ll -h features.csv 
ls -l -h features.csv 
wc -l all_controls.csv
ls -lh all_controls.csv 
vi all_controls.csv 
python features_cosFace.py 2D best_model_cosFace_acc2D.bin best_centers_cosFace_acc2D.bin  all_controls.csv 100
wc -l features.csv 
ls -lh features.csv
wc -l preds.csv 
vi preds.csv 
python features_cosFace.py 2D best_model_cosFace_acc2D.bin best_centers_cosFace_acc2D.bin  all_controls.csv 100
wc -l features.csv 
python features_cosFace.py 2D best_model_cosFace_acc2D.bin best_centers_cosFace_acc2D.bin  all_controls.csv 100
vi all_controls.csv 
python features_cosFace.py 2D best_model_cosFace_acc2D.bin best_centers_cosFace_acc2D.bin  all_controls.csv 100
python train_cosFace.py 2Db none none  all_controls.csv 100
python features_cosFace.py 2D best_model_cosFace_loss2Db.bin best_centers_cosFace_loss2Db.bin  all_controls.csv 100
python features_cosFace.py train1 best_model_cosFace_loss2Db.bin best_centers_cosFace_loss2Db.bin  my_train1.set 100
head my_train1.set
head all_controls.csv 
python features_cosFace.py train1 best_model_cosFace_loss2Db.bin best_centers_cosFace_loss2Db.bin  my_train1.set 100
python features_cosFace.py train2 best_model_cosFace_loss2Db.bin best_centers_cosFace_loss2Db.bin  my_train2.set 100
python features_cosFace.py train3 best_model_cosFace_loss2Db.bin best_centers_cosFace_loss2Db.bin  my_train3.set 100
python features_cosFace.py train3 best_model_cosFace_loss2Db.bin best_centers_cosFace_loss2Db.bin  my_train4.set 100
exit
sudo mount -o discard,defaults /dev/sdb /data
cd /data/code/cells/
ls -ltr
vi features_cosFace.py 
vi all_controls.csv 
vi features_cosFace.py 
vi cosface2D.py 
vi cosface.py 
vi cosface2D.py 
vi features_cosFace.py 
vi cosface2D.py 
vi cosface.py 
vi cosface2D.py 
ls -ltr | tail
top
df -h
ls -ltr | tail
vi features_cosFace.py 
top
vi features_cosFace.py 
vi features.csv 
wc -l features.csv 
wc -l all_controls.csv 
awk '{if (NR%2==1) print }' features.csv | wc -l
awk '{if (NR%2==1) print }' features.csv > f1.csv
awk '{if (NR%2==0) print }' features.csv > f2.csv
vi f1.csv 
ls *set
ls ../../input/
ls
vi my_train1.set
wc -l my_train1.set
ls -ltr | tail
vi features_cosFace.py 
vi cosface2D.py 
vi features_cosFace.py 
vi cosface2D.py 
vi features_cosFace.py 
ls -ltr | tail
mv preds.csv preds_controls.csv
mv features.csv features_controls.csv
vi ~/.bash_history 
vi all_controls.csv 
grep -P ",0$" all_controls.csv | less
vi train_controls.csv 
grep -P ",0$" all_controls.csv | less
grep -P ",30$" all_controls.csv | less
grep -P ",31$" all_controls.csv | less
mkdir controls
for f in 1 2; do grep -P ",$f$" all_controls.csv > controls/$f_controls.csv; done
ls controls/
for f in 1 2; do grep -P ",$f$" all_controls.csv > controls/${f}_controls.csv; done
ls controls/
ls -a controls/
rm controls/.csv 
head controls/*
for((f=0;f<=30;f++)); do grep -P ",$f$" all_controls.csv > controls/${f}_controls.csv; done
ls controls/
wc -l controls/*
vi ~/.bash_history 
ls -ltr | tail
wc -l features_train1.csv 
for f in $(awk -v FS="_" '{print $1"_"$2}' my_train1.set); do if [ "$oldF" != $f ]; then A=$(grep $f controls/0_controls.csv | head -n 1); fi; echo $A; oldF=$f; done | cut -f 4 -d, > a1
wc -l a1 my_train1.set 
head a1
mkdir tmp
for c in 1 2; do for f in $(awk -v FS="_" '{print $1"_"$2}' my_train1.set); do if [ "$oldF" != $f ]; then A=$(grep $f controls/${c}_controls.csv | head -n 1); fi; echo $A; oldF=$f; done | cut -f 4 -d, > tmp/a_$c; done
ls tmp/
for((c=0;c<=30;c++)); do for f in $(awk -v FS="_" '{print $1"_"$2}' my_train1.set); do if [ "$oldF" != $f ]; then A=$(grep $f controls/${c}_controls.csv | head -n 1); fi; echo $A; oldF=$f; done | cut -f 4 -d, > tmp/a_$c; echo $c; done
wc -l tmp/*
vi ~/.bash_history 
cp my_train1.set aaa; for((c=0;c<=30;c++)); do paste -d, aaa tmp/a_{$c} > bbb; mv bbb aaa; done
cp my_train1.set aaa; for((c=0;c<=30;c++)); do paste -d, aaa tmp/a_${c} > bbb; mv bbb aaa; done
wc -l aaa
vi aaa
ls -ltr | tail
vi aaa 
wc -l features_train1.csv 
awk '{if(NR%2==1) print }' features_train1.csv > features_train1_1.csv
awk '{if(NR%2==0) print }' features_train1.csv > features_train1_2.csv
wc -l features_train1_1.csv my_train1.set 
wc -l aaa
mv aaa my_train1_controls.set 
wc -l my_train1_controls.set
wc -l 
wc -l features_train1_1.csv
ls all_controls.csv 
wc -l all_controls.csv 
wc -l features_
awk '{if(NR%2==1) print }' features_controls.csv > features_controls_1.csv
awk '{if(NR%2==0) print }' features_controls.csv > features_controls_2.csv
less features_controls_1.csv
vi all_controls.csv 
vi my_train1.set 
vi my_train1_controls.set 
awk -v FS=, '{print $2"_"$3"_"$7 }' my_train1_controls.set | less
awk -v FS=, '{print $2"_"$3"_"$7 }' my_train1_controls.set | head
grep -n HUVEC-18_3_C07 all_controls.csv 
grep -n HUVEC-18_3_C07 all_controls.csv | cut -d: -f 1
wc -l features_train1_1.csv 
A=$(grep -n HUVEC-18_3_C07 all_controls.csv | cut -d: -f 1)
echo $A
echo $((A+1))
head all_controls.csv 
echo $((A-1))
head -n 685 features_train1_1.csv | tail -n 1 | wc -l
head -n 685 features_train1_1.csv | tail -n 1 | less
vi ~/.bash_history 
wc -l cc1
vi cc1 
awk -v FS=, '{prin NF}' cc1 | head
awk -v FS=, '{print NF}' cc1 | head
wc -l features_train1_1.csv 
wc -l cc1
exit
cd /data/code/cells/
ls -ltr
vi calculate_feature_cosines.sh
chmod +x calculate_feature_cosines.sh 
./calculate_feature_cosines.sh | wc -l
./calculate_feature_cosines.sh | less
date (./calculate_feature_cosines.sh > /dev/null);
date ./calculate_feature_cosines.sh | head -n 10 > /dev/null;
man time
man date
time ls
time ./calculate_feature_cosines.sh | head -n 10 > /dev/null;
time ./calculate_feature_cosines.sh | head -n 100 > /dev/null;
time ./calculate_feature_cosines.sh | head -n 200 > /dev/null;
vi calculate_feature_cosines.sh 
wc -l my_train1_controls.set
vi my_train1_controls.set
time ./calculate_feature_cosines.sh > cc1;
exit
cd /data/code/cells/
vi my_train1B.set 
vi ImagesDS_controls.py 
ls -ltr *py
cp train_regular.py train_controls.py
vi train_controls.py 
vi ImagesDS_controls.py 
vi train_controls.py 
cp DensNet.py DensNet_controls.py
vi DensNet_controls.py 
vi ~/.bash_history 
vi ImagesDS_controls.py 
vi DensNet_controls.py 
exit
sudo mount -o discard,defaults /dev/sdb /data
cd /data/code/cells/
ls -ltr
vi ~/.bash_history 
ls -l cc
vi calculate_feature_cosines.sh 
ls -ltr cc
ls -ltr
vi cc.1 
vi calculate_feature_cosines.sh 
vi calculate_cosines.awk 
vi calculate_feature_cosines.sh 
wc -l cc.1
wc -l cc/cc.8
vi cc/cc.8
vi cc.1 
awk '{print NF}' cc.1 | less
vi cc.1 
ls cc/cc*
paste cc* | less
paste cc/* | less
vi cc.1 
less cc.1 
wc -l my_train1.set 
wc -l my_train1_controls.set 
wc -l cc.1
vi cc.1 
for((i=1;i<=31;i++)); do echo C$i.1a C$i.1b C$i.2a C$i.2b; done | less
for((i=1;i<=31;i++)); do echo -n C$i.1a C$i.1b C$i.2a C$i.2b; done | less
for((i=1;i<=31;i++)); do echo -n C$i.1a C$i.1b C$i.2a C$i.2b ; done | less
for((i=1;i<=31;i++)); do echo -n "C$i.1a C$i.1b C$i.2a C$i.2b "; done | less
(for((i=1;i<=31;i++)); do echo -n "C$i.1a C$i.1b C$i.2a C$i.2b "; done; echo; cat cc.1) > cc.2
vi cc.2
vi my_train1_controls.set 
paste my_train1.set cc.2 > my_train1B.set
ll -h my_train1*
ls -l -h my_train1*
vi my_train1B.set
paste my_train1.set cc.2 -d"," > my_train1B.set
ls -l -h my_train1*
vi my_train1B.set
ls -ltr *py
vi train_regular.py 
cp ImagesDS.py ImagesDS_controls.py
vi ImagesDS_controls.py
(for((i=1;i<=31;i++)); do echo -n "self.records[myInd].C${i}_1a,self.records[myInd].C${i}_1b"; done; echo;) |  less
(for((i=1;i<=31;i++)); do echo -n "self.records[myInd].C${i}_1a,self.records[myInd].C${i}_1b,"; done; echo;) |  less
(for((i=1;i<=31;i++)); do echo -n "self.records[myInd].C${i}_2a,self.records[myInd].C${i}_2b,"; done; echo;) |  less
vi ImagesDS_controls.py
(for((i=1;i<=31;i++)); do echo -n "self.records[myInd].C${i}_1a,"; done; echo;) |  less
(for((i=1;i<=31;i++)); do echo -n "self.records[myInd].C${i}_2a,"; done; echo;) |  less
vi train_controls.py 
python train_controls.py CC1 final_model_512_21.bin my_train1B.set 30
exit
cd /data/code/cells/
vi train_controls.py 
vi ImagesDS_controls.py 
vi DensNet_controls.py 
vi ImagesDS_controls.py 
vi DensNet_controls.py 
vi train_controls.py 
vi ImagesDS_controls.py 
vi DensNet_controls.py 
vi train_controls.py 
vi DensNet_controls.py 
ls -ltr
vi train_controls.py 
wc -l cc.1
wc -l cc.2
vi cc.2
awk '{for(i=1;i<=NF/2;i++) printf("%f,", ($(2*i)+$(2*i-1))/2 ); print "";}' cc.2 | less
awk -v FS="," '{for(i=1;i<=NF/2;i++) printf("%f,", ($(2*i)+$(2*i-1))/2 ); print "";}' cc.2 | less
awk -v FS="," '{for(i=1;i<=NF/2;i++) printf("%f,", ($(2*i)+$(2*i-1))/2 ); print "";}' cc.2 | awk -v FS=, '{print NF}' | wc -l
awk -v FS="," '{for(i=1;i<=NF/2;i++) printf("%f,", ($(2*i)+$(2*i-1))/2 ); print "";}' cc.2 | awk -v FS=, '{print NF}' > cc
vi ~/.bash_history 
awk -v FS="," '{for(i=1;i<=NF/2;i++) printf("%f,", ($(2*i)+$(2*i-1))/2 ); print "";}' cc.2 > cc.3
vi cc.3
vi ~/.bash_history 
(for((i=1;i<=31;i++)); do echo -n "C${i}_1,C${i}_2,"; done; echo; cat cc.3) > cc.4
vi cc.4
vi ~/.bash_history 
paste my_train1.set cc.2 > my_train1C.set
vi ~/.bash_history 
(for((i=1;i<=31;i++)); do echo -n "self.records[myInd].C${i}_1a,"; done; echo;) |  less
(for((i=1;i<=31;i++)); do echo -n "self.records[myInd].C${i}_1,"; done; echo;) |  less
vi ImagesDS.py 
vi ImagesDS_controls.py 
(for((i=1;i<=31;i++)); do echo -n "self.records[myInd].C${i}_2,"; done; echo;) |  less
vi ImagesDS_controls.py 
vi my_train1C.set
vi cc.4
paste my_train1.set cc.4 -d, > my_train1C.set
vi my_train1C.set
vi my_train1B.set
vi my_train1C.set
vi ImagesDS_controls.py 
vi DensNet_controls.py 
sudo mount -o discard,defaults /dev/sdb /data
cd /data/code/cells/
python train_controls.py CC1 final_model_512_21.bin my_train1B.set 30
vi train_controls.py 
python train_controls.py CC1 final_model_512_21.bin my_train1B.set 30
vi train_controls.py 
python train_controls.py CC1 final_model_512_21.bin my_train1B.set 30
vi train_controls.py 
python train_controls.py CC1 final_model_512_21.bin my_train1B.set 30
vi train_controls.py 
python train_controls.py CC1 final_model_512_21.bin my_train1B.set 30
python train_controls.py CC1 final_model_512_21.bin my_train1C.set 30
sudo mount -o discard,defaults /dev/sdb /data
cd /data/code/cells/
python train_controls.py CC1 final_model_512_21.bin my_train1C.set 30
ls =ltr ) tail
ls -ltr |  tail
python train_controls.py CC1 best_model_regular_lossCC1.bin my_train1C.set 30
screen -R regular
exit
cd /data/code/cells/
vi DensNet_controls.py 
vi train_controls.py 
tail train.out 
exit
cd /data/code/cells/
ls -ltr | tail
vi train.out 
exit
python train_controls.py CC1 best_model_regular_lossCC1.bin my_train1C.set 30
python train_controls.py CC1 best_model_regular_lossCC1.bin my_train1C.set 30 > train.out
sudo mount -o discard,defaults /dev/sdb /data
cd /data/code/cells/
ls -ltr
ls -ltr *py
cp train_controls.py predict_controls.py
vi predict_controls.py 
cd /data/code/cells/
vi my_train1.set 
cd /data/code/cells/
vi predict_controls.py 
ls tmp/
less tmp/a_0
vi prepare_data.sh
wc -l test1.set 
vi prepare_data.sh
chmod +x prepare_data.sh 
./prepare_data.sh 
vi prepare_data.sh
./prepare_data.sh 
ls -ltr | tail
vi test1_controls.set 
vi calculate_feature_cosines.sh 
top
ls
ls -ltr
vi prepare_data.sh 
vi ~/.bash_history 
ls -l cc
less cc/cc.9
vi prepare_data.sh 
./prepare_data.sh 
ls -ltr | tail
wc -l cc*
vi cc.4 
vi prepare_data.sh 
ls cc/
vi cc/cc.1
vi cc/cc.8
vi prepare_data.sh 
./prepare_data.sh 
vi cc.4 
vi prepare_data.sh 
./prepare_data.sh 
vi prepare_data.sh 
vi cc.4 
wc -l test1.set 
wc -l cc.4
wc -l cc/cc.1
wc -l cc/cc.8
exit
cd /data/code/cells/
sudo mount -o discard,defaults /dev/sdb /data
cd /data/code/cells/
python predict_controls.py CC1 best_model_regular_lossCC1.bin test1.set 
vi test1.set 
vi ~/.bash_history 
vi ./calculate_feature_cosines.sh
less my_train1_controls.set
vi ~/.bash_history 
vi calculate_feature_cosines.sh 
./calculate_feature_cosines.sh test1_controls.set 
wc -l cc.*
vi calculate_feature_cosines.sh 
./calculate_feature_cosines.sh test1_controls.set 
wc -l cc/cc.6
wc -l test1_controls.set
vi calculate_feature_cosines.sh 
wc -l my_train1*
ls -l cc/cc.6
vi calculate_feature_cosines.sh 
wc -l features_train1_1.csv
vi ~/.bash_history 
exit
cd /data/code/cells/
vi calculate_feature_cosines.sh 
sudo mount -o discard,defaults /dev/sdb /data
cd /data/code/cells/
ls features_*
cd /data/code/cells/
vi ImagesDS.py 
vi features_cosFace.py 
vi ImagesDS.py 
vi features_cosFace.py 
vi ImagesDS
vi ImagesDS.py 
vi calculate_feature_cosines.sh 
python features_cosFace.py test2 best_model_cosFace_loss2Db.bin best_centers_cosFace_loss2Db.bin  test2.set test; python features_cosFace.py test3 best_model_cosFace_loss2Db.bin best_centers_cosFace_loss2Db.bin  test3.set test; python features_cosFace.py test4 best_model_cosFace_loss2Db.bin best_centers_cosFace_loss2Db.bin  test4.set test
vi predict_controls.py 
vi train.out 
vi ~/.bash_history 
vi train_controls.
vi train_controls.py 
vi train_controls.
vi train_controls.py 
vi DensNet_controls.py 
python train_controls.py Lin1 none my_train1C.set 30
vi DensNet_controls.py 
python train_controls.py Lin1 none my_train1C.set 30
screen -R regular
screen -r regular
top
ls -ltr *py
vi train_cosFace.py 
ls -ltr
vi train_linear.out 
ls -ltr
git status
git diff
exit
sudo mount -o discard,defaults /dev/sdb /data
cd /data/code/cells/
python features_cosFace.py test1 best_model_cosFace_loss2Db.bin best_centers_cosFace_loss2Db.bin  test1.set 100
python features_cosFace.py test1 best_model_cosFace_loss2Db.bin best_centers_cosFace_loss2Db.bin  test1.set test
vi ~/.bash_history 
ls features_*csv
awk '{if(NR%2==1) print }' features_train1.csv > features_train1_1.csv
awk '{if(NR%2==0) print }' features_test1.csv > features_test1_2.csv
awk '{if(NR%2==1) print }' features_test1.csv > features_test1_1.csv
vi calculate_feature_cosines.sh 
./calculate_feature_cosines.sh test1_controls.set features_test1_1.csv features_test1_2.csv
wc -l cc/cc.6
wc -l test1_controls.set
less test1_controls.set
less cc/cc.6
vi prepare_data.sh 
./prepare_data.sh 
ls -ltr | tail
vi prepare_data.sh 
wc -l test1.set cc.4
vi cc.4
vi prepare_data.sh 
./prepare_data.sh 
ls -ltr | tail
vi test1B.set
python predict_controls.py CC1 best_model_regular_lossCC1.bin test1B.set
vi predict_controls.py 
vi ImagesDS.py 
vi ImagesDS_controls.py 
python predict_controls.py CC1 best_model_regular_lossCC1.bin test1B.set
vi predict_controls.py 
python predict_controls.py CC1 best_model_regular_lossCC1.bin test1B.set
vi predict_controls.py 
python predict_controls.py CC1 best_model_regular_lossCC1.bin test1B.set
vi predict_controls.py 
python predict_controls.py CC1 best_model_regular_lossCC1.bin test1B.set
vi test1B.set
vi test1.set
python predict_controls.py CC1 best_model_regular_lossCC1.bin test1B.set
ls -ltr
vi tmp.csv 
grep HEPG2-08_4 tmp.csv | less
grep HEPG2-08_4 tmp.csv | cut -f 2 -d, | less
grep HEPG2-08_4 tmp.csv | cut -f 2 -d, | sort | uniq | wc -l
vi sirnas.1 
grep HEPG2-08_4 tmp.csv | cut -f 2 -d, | sort | uniq | sort -n > a1
meld a1 sirnas.1
diff a1 sirnas.1
comm -1 -2 sirnas.1 a1 | wc -l
grep HEPG2-08_4 tmp.csv | cut -f 2 -d, | sort | uniq > a1
sort sirnas.1 > a2
comm -1 -2 a2 a1 | wc -l
comm -1 -3 a2 a1 | wc -l
comm -3 -2 a2 a1 | wc -l
vi sirnas.1 
python predict_controls.py CC1 best_model_regular_lossCC1.bin test1B.set sirnas.1
ls -ltr 
python predict_controls.py CC1 best_model_regular_lossCC1.bin test1B.set sirnas.1
vi tmp.csv 
grep HEPG2-08_4 tmp.csv | cut -f 2 -d, | sort | uniq > a1
comm -1 -2 a1 a2 | wc -l
python predict_controls.py CC1 best_model_regular_lossCC1.bin test1B.set sirnas.1
grep HEPG2-08_4 tmp.csv | cut -f 2 -d, | sort | uniq > a1
comm -1 -2 a1 a2 | wc -l
python predict_controls.py CC1 best_model_regular_lossCC1.bin test1B.set sirnas.1
grep HEPG2-08_4 tmp.csv | cut -f 2 -d, | sort | uniq > a1
comm -1 -2 a1 a2 | wc -l
python predict_controls.py CC1 best_model_regular_lossCC1.bin test1B.set sirnas.1 submssion1n.csv
ls -ltr | tail
cut -d, -f 2 submssion1n.csv | less
cut -d, -f 2 submssion1n.csv | sort | uniq -c | sort -rnk1 | less
python predict_controls.py CC1 best_model_regular_lossCC1.bin test1B.set sirnas.1 submssion1n.csv
cut -d, -f 2 submssion1n.csv | sort | uniq -c | sort -rnk1 | less
vi prepare_data.sh 
vi test_controls.csv 
 all_controls.csv 
vi prepare_data.sh 
ls -ltr features_*
vi calculate_cosines.awk 
vi calculate_feature_cosines.sh 
wc -l features_controls_2.csv 
wc -l features_controls_1.csv 
less features_controls_1.csv 
ls -ltr *py
vi features_cosFace.py
vi preds_controls.csv 
wc -l preds_controls.csv 
wc -l all_controls.csv 
vi all_controls.csv 
pwd
git status
git add *py
git status | less
vi .gitignore 
git status 
git add calculate_cosines.awk calculate_feature_cosines.sh prepare_data.sh zero.row
git status
rm cc.* a1 a2 cc1 cc2
git status
vi .gitignore 
vi negative.controls 
git status
vi .gitignore 
git status
git config --global core.editor "vim"
git commit -a
git push
git status
python predict_controls.py CC1 best_model_regular_lossCC1.bin my_train1C.set sirnas.1 tmp1.csv
wc -l tmp1.csv 
vi tmp1.csv 
exit
cd /data/code/cells/
ls -ltr | tail
vi train_linear.out 
ls *out
vi train.out 
vi train_linear.out 
ls -ltr | tail
vi train_linear.out 
ls -ltr | tail
vi train_linear.out 
ls -ltr | tail
vi train_linear.out 
ls -ltr | tail
vi train_linear.out 
ls -ltr | tail
vi train_linear.out 
exit
python train_controls.py Lin1 none my_train1C.set 30 > train_linear.out 
python train_controls.py Lin2 best_model_regular_lossLin1.bin my_train1C.set 30
exit
cd /data/code/cells/
ls -ltr | tail
vi train_linear.out 
ls -ltr
vi DensNet_controls.py 
vi ImagesDS_controls.py 
vi train_controls.py 
vi train_linear.out 
vi train_controls.py 
screen -r regular
vi ImagesDS_controls.py 
screen -r regular
vi DensNet_controls.py 
screen -r regular
vi my_train1C.set 
cut -f 5 -d, my_train1C.set | less
cut -f 4 -d, my_train1C.set | less
cut -f 4 -d, my_train1C.set | sort | uniq -c | less
cut -f 4,5 -d, my_train1C.set | less
cut -f 4,5 -d, my_train1C.set | grep 1074
cut -f 4,5 -d, my_train1C.set | grep -P ",12$"
cut -f 4,5 -d, my_train1C.set | grep -P ",14$"
cut -f 4,5 -d, my_train1C.set | grep -P ",16$"
cut -f 4,5 -d, my_train1C.set | less
cut -f 4,5 -d, my_train1C.set | grep -P ",53$"
vi my_train1C.set 
vi my_train1B.set 
vi my_train1.set 
screen -r regular
vi train_linear.out 
screen -r regular
vi my_train1.set 
vi my_train1C.set 
grep ",1007," my_train1C.set | less
grep ",1007," my_train1C.set | cut -f 8 -d, | less
grep ",1007," my_train1C.set | cut -f 38 -d, | less
screen -r regular
ls -ltr | tail
vi ImagesDS_controls.py 
ls -ltr | tail
screen -r regular
ls -ltr | tail
screen -r regular
vi train_linear2.out 
screen -r regular
exit
sudo mount -o discard,defaults /dev/sdb /data
cd /data/code/cells/
ls -ltr
vi train_linear2.out 
vi my_train1.set 
grep U2OS my_train1.set | wc -l
grep U2OS my_train1C.set > my_train1C_U2OS.set
vi my_train1C.set
vi my_train1C_U2OS.set 
cut -f 1 -d, my_train1C_U2OS.set | less
cut -f 1 -d, my_train1C_U2OS.set > aa
vi aa
rm aa
vi train_controls.py 
python train_controls.py Lin2_U2OS best_model_regular_lossLin2.bin my_train1C_U2OS.set 30
python train_controls.py Lin2_U2OS best_model_regular_lossLin2.bin my_train1C_U2OS.set 30 | less
python train_controls.py Lin2_U2OS best_model_regular_lossLin2.bin my_train1C_U2OS.set 30 
exit
cd /data/code/cells/
vi ImagesDS_controls.py 
vi train_controls.py 
vi ImagesDS_controls.py 
ls ../../input/
cat ../../input/test.folders 
cat ../../input/train.folders 
vi ImagesDS_controls.py 
vi train_controls.py 
vi features_train1_1.csv 
wc -l features_train1_1.csv 
vi train_controls.py 
exit
sudo mount -o discard,defaults /dev/sdb /data
cd /data/code/cells/
ls -ltr
vi train_controls.py 
vi ImagesDS_controls.py 
vi my_train1C.set 
(head -n 1 my_train1C.set; grep HUVEC my_train1C.set ) > my_train1C_HUVEC.set
wc -l my_train1C_HUVEC.set
vi my_train1C_HUVEC.set 
vi train_controls.py 
python train_controls.py Lin2_HUVEC best_model_regular_lossLin2.bin my_train1C_HUVEC.set 100 
vi train_controls.py 
screen -list
screen -R regular
ls -ltr
vi train_controls.py 
screen -r regular
top
screen -r regular
exit
cd /data/code/cells/
ls -ltr
vi train_HUVEC.out 
ls -ltr | tail
vi train_HUVEC.out 
ls -ltr | tail
vi train_HUVEC.out 
ls -ltr | tail
vi train_HUVEC.out 
ls -ltr | tail
vi train_HUVEC.out 
screen -r regular
vi DensNet_controls.py 
grep zeros *py
vi DensNet_controls.py 
screen -r regular
vi train_HUVEC.out 
screen -r regular
diff DensNet_controls.py DensNet_negative.py 
diff DensNet_controls.py DensNet_negative.py | less
screen -r regular
vi DensNet_controls.py 
vi train_controls.py 
vi DensNet_controls.py 
screen -r regular
vi DensNet_controls.py 
screen -r regular
vi DensNet_controls.py 
screen -r regular
vi DensNet_controls.py 
screen -r regular
vi DensNet_controls.py 
vi train_controls.py 
screen -r regular
vi train_controls.py 
screen -r regular
diff DensNet_negative.py DensNet_controls.py 
vi DensNet_negative.py 
vi DensNet_controls.py 
vi DensNet_negative.py 
diff ImagesDS_controls.py ImagesDS_negative.py 
vi DensNet_negative.py 
diff train_controls.py train_negative.py 
vi train_negative.py 
diff train_controls.py train_negative.py 
vi train_negative.py 
diff train_controls.py train_negative.py 
vi train_negative.py 
diff -w train_controls.py train_negative.py 
vi train_negative.py 
diff -w train_controls.py train_negative.py 
screen -r regular
vi DensNet_negative.py 
vi train_controls.py 
vi DensNet_negative.py 
vi train_negative.py 
screen -r regular
vi ImagesDS_negative.py 
vi my_train1N.set 
vi my_train1C_HUVEC.set
(head -n 1 my_train1N.set; grep HUVEC my_train1N.set ) > my_train1N_HUVEC.set
screen -r regular
vi train_negative.py 
wc -l my_train1N_HUVEC.set
screen -r regular
exit
cd /data/code/cells/
ls -ltr
vi train_negative.out 
screen -r regular
vi train_negative.py 
vi ImagesDS_negative.py 
vi train_negative.py 
vi ImagesDS_negative.py 
vi ImagesDS.py 
vi ImagesDS_negative.py 
vi DensNet_negative.py 
vi train_negative.py 
vi DensNet_negative.py 
screen -r regular
less my_train1N_HUVEC.set
screen -r regular
less my_train1N_HUVEC.set
vi train_negative.out
vi DensNet_negative.py 
screen -r regular
vi DensNet_negative.py 
screen -list
exit
screen -r regular
exit
screen -r regular
cd /data/code/cells/
vi ImagesDS_negative.py 
ls -ltr *out
vi train_HUVEC.out 
vi DensNet_negative.py 
cp ImagesDS_negative.py ImagesDS_pControls.py
vi ImagesDS_pControls.py 
exit
cd /data/code/cells/
screen -r regular
vi myDensenets.py
vi DensNet_negative.py 
vi myDensenets.py
vi DensNet_negative.py 
screen -r regular
vi DensNet_negative.py 
screen -r regular
vi DensNet_negative.py 
screen -r regular
vi myDensenets.py
screen -r regular
vi myDensenets.py
screen -r regular
vi DensNet_negative.py 
screen -r regular
python train_negative.py Lin2_HUVECn none my_train1N_HUVEC.set 100
vi DensNet_negative.py 
python train_negative.py Lin2_HUVECn none my_train1N_HUVEC.set 100
vi DensNet_negative.py 
vi DensNet_controls.py 
vi DensNet_negative.py 
python train_negative.py Lin2_HUVECn none my_train1N_HUVEC.set 100
vi DensNet_negative.py 
vi ImagesDS_negative.py 
vi DensNet_negative.py 
vi ImagesDS_negative.py 
vi DensNet_negative.py 
python train_negative.py Lin2_HUVECn none my_train1N_HUVEC.set 100
vi DensNet_negative.py 
vi ImagesDS_negative.py 
python train_negative.py Lin2_HUVECn none my_train1N_HUVEC.set 100
vi DensNet_negative.py 
python train_negative.py Lin2_HUVECn none my_train1N_HUVEC.set 100
vi DensNet_negative.py 
python train_negative.py Lin2_HUVECn none my_train1N_HUVEC.set 100
vi DensNet_negative.py 
python train_negative.py Lin2_HUVECn none my_train1N_HUVEC.set 100
vi DensNet_negative.py 
python train_negative.py Lin2_HUVECn none my_train1N_HUVEC.set 100
vi cosface2D.py 
vi ImagesDS_negative.py 
python train_negative.py Lin2_HUVECn none my_train1N_HUVEC.set 100
vi ImagesDS_negative.py 
python train_negative.py Lin2_HUVECn none my_train1N_HUVEC.set 100
vi ImagesDS_negative.py 
python train_negative.py Lin2_HUVECn none my_train1N_HUVEC.set 100
vi ImagesDS_negative.py 
vi DensNet_negative.py 
python train_negative.py Lin2_HUVECn none my_train1N_HUVEC.set 100
vi DensNet_negative.py 
python train_negative.py Lin2_HUVECn none my_train1N_HUVEC.set 100
vi DensNet_negative.py 
python train_negative.py Lin2_HUVECn none my_train1N_HUVEC.set 100
vi DensNet_negative.py 
vi train_negative.py 
python train_negative.py Lin2_HUVECn none my_train1N_HUVEC.set 100
vi train_negative.py 
python train_negative.py Lin2_HUVECn none my_train1N_HUVEC.set 100
vi train_negative.py 
python train_negative.py Lin2_HUVECn none my_train1N_HUVEC.set 100
ls -ltr | tail
screen -r regular
vi DensNet_negative.py 
screen -r regular
vi ImagesDS_negative.py 
screen -r regular
vi ImagesDS_negative.py 
screen -r regular
vi ImagesDS_negative.py 
python train_negative.py Lin2_HUVECn none my_train1N_HUVEC.set 100
screen -r regular
vi ImagesDS_negative.py 
screen -r regular
exit
screen -r regular
python train_controls.py Lin2_HUVEC best_model_regular_lossLin2.bin my_train1C_HUVEC.set 100
python train_controls.py Lin2_HUVEC best_model_regular_lossLin2.bin my_train1C_HUVEC.set 100 > train_HUVEC.out
python train_controls.py Lin2_HUVECx best_model_regular_lossLin2_HUVEC.bin my_train1C_HUVEC.set 100 
python train_controls.py Lin2_HUVECx best_model_regular_lossLin2.bin my_train1C_HUVEC.set 100 
python train_controls.py Lin2_HUVECx best_model_regular_lossLin.bin my_train1C_HUVEC.set 100 
python train_controls.py Lin2_HUVECx best_model_regular_lossLin1.bin my_train1C_HUVEC.set 100 
python train_controls.py Lin2_HUVECx none my_train1C_HUVEC.set 100 
python train_controls.py Lin2_HUVECy none my_train1C_HUVEC.set 100 
python train_negative.py Lin2_HUVECn none my_train1C_HUVEC.set 100 
python train_negative.py Lin2_HUVECn none my_train1N_HUVEC.set 100 
python train_negative.py Lin2_HUVECn none my_train1N_HUVEC.set 100 > train_negative.out
python train_negative.py Lin2_HUVECn none my_train1N_HUVEC.set 100 
:q
python train_negative.py Lin2_HUVECn none my_train1N_HUVEC.set 100 
python train_negative.py Lin2_HUVECn2 best_model_regular_lossLin2_HUVECn.bin my_train1N_HUVEC.set 100 
python train_negative.py Lin2_HUVECn2 none my_train1N_HUVEC.set 100 
sudo mount -o discard,defaults /dev/sdb /data
cd /data/code/cells/
ls -ltr | tail
python train_negative.py Lin2_HUVECn2 best_model_regular_lossLin2_HUVECn2.bin my_train1N_HUVEC.set 100 
vi train_negative.py 
vi ImagesDS_negative.py 
vi train_negative.py 
exit
cd /data/code/cells/
python train_negative.py Lin2_HUVECn2 none my_train1N_HUVEC.set 100 
vi DensNet_negative.py 
python train_negative.py Lin2_HUVECn2 none my_train1N_HUVEC.set 100 
python train_negative.py Lin2_HUVECn3 none my_train1N_HUVEC.set 100 
(head -n 1 train_controls.csv; grep HUVEC train_controls.csv ) > trainH_controls.csv
less trainH_controls.csv
python train_pControls.py pCon1 best_model_regular_lossLin2_HUVECn3.bin my_train1N_HUVEC.set trainH_controls.csv 100 
ls -ltr
vi tmp_HUVEC-01_B02_1_1138 
python train_pControls.py pCon1 best_model_regular_lossLin2_HUVECn3.bin my_train1N_HUVEC.set trainH_controls.csv 100 
vi tmp_HUVEC-01_B02_1_1138 
python train_pControls.py pCon1 best_model_regular_lossLin2_HUVECn3.bin my_train1N_HUVEC.set trainH_controls.csv 100 
vi tmp_HUVEC-01_B02_1_1138 
python train_pControls.py pCon1 best_model_regular_lossLin2_HUVECn3.bin my_train1N_HUVEC.set trainH_controls.csv 100 
ls controlFeatures/ | wc -l
ls controlFeatures/ | less
less controlFeatures/HUVEC-01_B02_1_1_1138
head controlFeatures/HUVEC-01_B02_1_1_1138
head controlFeatures/HUVEC-01_B02_1_2_1138
python train_pControls.py pCon1 best_model_regular_lossLin2_HUVECn2.bin my_train1N_HUVEC.set trainH_controls.csv 100 
head controlFeatures/HUVEC-01_B02_1_1_1138
head controlFeatures/HUVEC-01_B02_1_2_1138
ls -ltr
vi DensNet_negative.py 
python train_negative.py Lin2_HUVECn2B best_model_regular_lossLin2_HUVECn2.bin my_train1N_HUVEC.set 100 
python train_negative.py Lin2_HUVECn3B best_model_regular_lossLin2_HUVECn3.bin my_train1N_HUVEC.set 100 
python train_negative.py HnoC none my_train1N_HUVEC.set 100 
screen -list
screen -r regular
screen -R regular
exit
cd /data/code/
cd cells/
ls -ltr *py
vi ImagesDS_pControls.py 
vi all_controls.csv 
vi all_controls2.csv 
vi all_controls.csv 
head all_controls.csv 
head train_controls.csv 
vi ~/.bash_history 
head all_controls.csv 
head all_controls2.csv 
head train_controls.csv 
grep HEPG2-08_1_B02 all_controls2.csv | head
wc -l all_controls*
ls train_controls.csv 
ll train_controls.csv 
ls -l train_controls.csv 
head train_controls.csv
ls -ltr *py
cp train_negative.py train_pControls.py
vi train_pControls.py 
ls ../../input/train.folders 
cat ../../input/train.folders 
vi train_pControls.py 
vi ImagesDS_pControls.py 
vi train_pControls.py 
python train_pControls.py Lin2_HUVECn2 none my_train1N_HUVEC.set 100 
vi train_pControls.py 
vi ~/.vimrc 
vi train_pControls.py 
python train_pControls.py Lin2_HUVECn2 none my_train1N_HUVEC.set 100 
vi train_pControls.py 
python train_pControls.py Lin2_HUVECn2 none my_train1N_HUVEC.set 100 
vi train_pControls.py 
python train_pControls.py Lin2_HUVECn2 none my_train1N_HUVEC.set 100 
vi train_pControls.py 
vi ImagesDS_controls.py 
vi ImagesDS_negative.py 
grep randint *py
vi DensNet_negative.py 
vi best_learningHUVEC.out
ls -ltr
vi best_learningHUVEC.out
vi DensNet_negative.py 
vi best_learningHUVEC.out
vi DensNet_negative.py 
vi best_learningHUVEC.out
vi ImagesDS_negative.py 
vi best_learningHUVEC.out
vi DensNet_negative.py 
vi best_learningHUVEC.out
vi best2_learningHUVEC.out
vi DensNet_negative.py 
ls -ltr *py
vi train_pControls.py 
less all_controls.csv 
less train_controls.csv 
vi train_pControls.py 
ls *py
vi predict_controls.py 
vi train_pControls.py 
vi predict_controls.py 
vi train_pControls.py 
vi ImagesDS_controls.py 
vi train_pControls.py 
vi predict_controls.py 
vi train_pControls.py 
vi ImagesDS_controls.py 
vi train_pControls.py 
vi ImagesDS_controls.py 
vi train_pControls.py 
vi best2_learningHUVEC.out 
vi withoutRelu_learningHUVEC.out 
ls -ltr *bin
ls -ltr *py

ls -ltr *csv
vi trainH_controls.csv
vi train_pControls.py 
ls -ltr *py
cp DensNet_negative.py DensNet_pControls.py
vi DensNet_pControls.py
vi train_pControls.py 
vi DensNet_pControls.py
vi train_pControls.py 
vi DensNet_pControls.py
vi train_pControls.py 
vi DensNet_pControls.py
vi train_pControls.py 
ls controls/
mkdir controlFeatures
vi train_pControls.py 
ls controlFeatures/
vi train_pControls.py 
ls -ltr | tail
wc -l trainH_controls.csv
vi train_pControls.py 
ls controlFeatures/ | wc -l
ls -ltr controlFeatures/
ls controlFeatures/ | wc -l
vi train_pControls.py 
vi ImagesDS_negative.py 
vi train_pControls.py 
ls controlFeatures/ | less
vi train_pControls.py 
vi ImagesDS_negative.py 
vi train_pControls.py 
vi DensNet_pControls.py 
vi train_pControls.py 
vi ImagesDS_negative.py 
vi train_pControls.py 
vi DensNet_pControls.py 
vi train_pControls.py 
rm controlFeatures/*
vi ImagesDS_negative.py 
vi train_pControls.py 
R
vi best2_learningHUVEC.out 
ls -ltr
vi best2_learningHUVEC.out 
vi train_negative.py 
ls -ltr | tail
less trainH_controls.csv
vi trainH_controls_min.csv
vi train_pControls.py 
mkdir controlFeatures1/
python train_pControls.py pCon1 best_model_regular_lossLin2_HUVECn2.bin my_train1N_HUVEC.set trainH_controls_min.csv 100
vi train_pControls.py 
vi DensNet_negative.py 
ls -ltr *bin | tail
cp best_model_regular_accLin2_HUVECn2B.bin model2.bin
cp best_model_regular_accLin2_HUVECn2.bin model1.bin
vi best2_learningHUVEC.out 
ls -ltr *bin | tail
cp best_model_regular_accLin2_HUVECn2B.bin model5.bin
cp best_model_regular_accLin2_HUVECn2B.bin model6.bin
vi best_learningHUVEC.out 
vi best2_learningHUVEC.out 
cp best_model_regular_accLin2_HUVECn2B.bin model9.bin
ls -ltr *bin | tail
vi best2_learningHUVEC.out 
ls -ltr *bin | tail
cp best_model_regular_accLin2_HUVECn2B.bin modelX.bin
vi best2_learningHUVEC.out 
vi best3_learningHUVEC.out 
ls -ltr out
ls -ltr *out
python train_pControls.py pCon1 model1.bin my_train1N_HUVEC.set trainH_controls_min.csv 100
python train_pControls.py pCon1 model2.bin my_train1N_HUVEC.set trainH_controls_min.csv 100
python train_pControls.py pCon1 model3.bin my_train1N_HUVEC.set trainH_controls_min.csv 100
ls -ltr model*
python train_pControls.py pCon1 model5.bin my_train1N_HUVEC.set trainH_controls_min.csv 100
python train_pControls.py pCon1 model6.bin my_train1N_HUVEC.set trainH_controls_min.csv 100
python train_pControls.py pCon1 model9.bin my_train1N_HUVEC.set trainH_controls_min.csv 100
python train_pControls.py pCon1 modelX.bin my_train1N_HUVEC.set trainH_controls_min.csv 100
diff modelX.bin model9.bin
vi train_pControls.py 
python train_pControls.py pCon1 modelX.bin my_train1N_HUVEC.set trainH_controls_min.csv 100
vi train_pControls.py 
python train_pControls.py pCon1 model1.bin my_train1N_HUVEC.set trainH_controls_min.csv 100
python train_pControls.py pCon1 model2.bin my_train1N_HUVEC.set trainH_controls_min.csv 100
python train_pControls.py pCon1 model5.bin my_train1N_HUVEC.set trainH_controls_min.csv 100
python train_pControls.py pCon1 model6.bin my_train1N_HUVEC.set trainH_controls_min.csv 100
python train_pControls.py pCon1 model9.bin my_train1N_HUVEC.set trainH_controls_min.csv 100
python train_pControls.py pCon1 modelX.bin my_train1N_HUVEC.set trainH_controls_min.csv 100
python train_pControls.py pCon1 model2.bin my_train1N_HUVEC.set trainH_controls_min.csv 100
vi DensNet_negative.py 
vi withoutRelu_learningHUVEC.out 
vi train_negative.py 
ls -ltr *py | tail
vi withoutRelu_learningHUVEC.out 
ls -ltr
vi train_HUVEC.out 
vi ~/.bash_history 
vi train.out
vi withoutRelu_learningHUVEC.out 
vi train_negative.py 
vi ImagesDS_negative.py 
vi DensNet_negative.py 
grep BOTH *negative.py
vi DensNet_negative.py 
vi ~/.bashrc 
wc -l ~/.bash_history 
vi ~/.bashrc 
exit
cd /data/code/cells/
ls
vi ~/.bash_history 
vi ~/.bashrc 
exit
exit
screen -r regular
ls -ltr | tail
cd /data/code/cells/
ls -ltr | tail
vi train_HUVEC_noCOntrols.out
cd /data/code/cells/
vi train_HUVEC_noCOntrols.out
ls -ltr *out
vi withoutRelu_learningHUVEC.out 
vi best2_learningHUVEC.out
vi train_HUVEC_noCOntrols.out
screen -r regular
ls -ltr *py
screen -r regular
vi train_negative.py 
vi DensNet_negative.py 
screen -r regular
vi train_HUVEC_noCOntrols.out
screen -r regular
vi train_HUVEC_noCOntrols.out
screen -r regular
vi DensNet_negative.py 
vi train_negative.py 
screen -r regular
vi DensNet_negative.py 
screen -r regular
ls -ltr
tail -f train_HUVEC_noCOntrols_Pretrained.out
vi DensNet_negative.py 
vi train_negative.
vi train_negative.py 
screen -r regular
tail -f train_HUVEC_noCOntrols_Pretrained.out
screen -r regular
ls -ltr | tail
tail -f train_HUVEC_noCOntrols_Pretrained.out
ls -ltr *out
vi train.out 
ls -ltr *log
ls -ltr 
vi train_linear2.out 
ls -ltr *out
vi train_HUVEC.out 
tail -f train_HUVEC_noCOntrols_Pretrained.out
ls -ltr *out
cat train_HUVEC_noCOntrols.out
tail -f train_HUVEC_noCOntrols_Pretrained.out
screen -r regular
exit
python train_negative.py HnoC none my_train1N_HUVEC.set 100
python train_negative.py HnoC none my_train1N_HUVEC.set 100 > train_HUVEC_noCOntrols.out
python train_negative.py HnoC_pT none my_train1N_HUVEC.set 100 > train_HUVEC_noCOntrols.out
python train_negative.py HnoC_pT none my_train1N_HUVEC.set 100 > train_HUVEC_noCOntrols_Pretrained.out
sudo mount -o discard,defaults /dev/sdb /data
cd /data/code/cells/
ls -ltr
exit
sudo mount -o discard,defaults /dev/sdb /data
cd /data/code/cells/
ls -ltr
vi ImagesDS_negative.py 
screen -r regular
screen -R regular
vi train_negative.
vi train_negative.py 
screen -R regular
ls -ltr
python train_negative.py HnoC_pT none my_train1N_HUVEC.set 100 > train_HUVEC_noCOntrols_noPretrained_noNormalize.out
screen -R regular
ls -ltr | tail
tail -f train_HUVEC_noCOntrols_noPretrained_noNormalize.out
top
ls -ltr | tail
python train_negative.py HnoC_noNorm none my_train1N_HUVEC.set 100 > train_HUVEC_noCOntrols_noPretrained_noNormalize.out
sudo mount -o discard,defaults /dev/sdb /data
cd /data/code/cells/
ls -ltr
screen -R regular
ls -ltr
tail -f train_HUVEC_noCOntrols_noPretrained_noNormalize.out
vi train_HUVEC_noCOntrols.out 
vi train_HUVEC_noCOntrols_Pretrained.out
exit
tail -f train_HUVEC_noCOntrols_noPretrained_noNormalize.out
git status
git pull
vi convert.py 
vi train_controls.py 
(head -n 1 train_controls.py; grep B02 train_controls.py) | wc -l
(head -n 1 train_controls.py; grep B02 train_controls.csv) | wc -l
(head -n 1 train_controls.py; grep B02 train_controls.csv) > negative.controls.csv
vi negative.controls
mv negative.controls.csv train.controls.csv
vi train.controls.csv
man split
split -l 33 --numeric-suffixes=1 train.controls.csv train.controls.csv.
ls train.controls.csv*
wc -l train.controls.csv*
ls -ltr | tain
ls -ltr | tail
mkdir new_train
python convert.py train.controls.csv.01 train new_train/
vi convert.py 
vi ImagesDS.py 
vi convert.py 
python convert.py train.controls.csv.01 train new_train/
vi convert.py 
python convert.py train.controls.csv.01 train new_train/
wc -l train_controls.csv 
for f in train.controls.csv.*; do (head -n 1 train_controls.csv; cat $f) > aaa; mv aaa $f; done 
vi train.controls.csv.01
python convert.py train.controls.csv.01 train new_train/
vi convert.py 
vi ImagesDS.py 
vi convert.py 
python convert.py train.controls.csv.01 train new_train/
vi convert.py 
python train_negative.py HnoC_noNorm none my_train1N_HUVEC.set 100 > train_HUVEC_noCOntrols_noPretrained_noNormalize.out
screen -r regular
python convert.py train.controls.csv.01 train new_train/
vi convert.py 
python convert.py train.controls.csv.01 train new_train/
vi convert.py 
ls ../../input/train/HEPG2-01/Plate1/B02_s1_w1.png
vi convert.py 
python convert.py train.controls.csv.01 train new_train/
vi convert.py 
screen -R c1
screen -R c2
screen -R c3
screen -R c4
screen -r c3
top
ls new_train/
ls new_train/*jpg | wc -l
find new_train/ -name "*png" | wc -l; date
find ../../input/train -name "*png" | wc -l; date
top
python convert.py train.controls.csv.04 train new_train/
exit
screen -r c4
python convert.py train.controls.csv.03 train new_train/
exit
screen -r c3
python convert.py train.controls.csv.02 train new_train/
exit
screen -r c2
python convert.py train.controls.csv.01 train new_train/
exit
screen -r c1
screen -list
screen -r regular
sudo mount -o discard,defaults /dev/sdb /data
cd /data/code/cells/
split -l 9 --numeric-suffixes=1 train.controls.csv train.controls.csv.
ls -ltr 
wc -l train.controls.csv.*
for f in train.controls.csv.* do; (head -n 1 train_controls.csv; cat $f) > aaa; mv aaa $f; done
for f in train.controls.csv.*; do (head -n 1 train_controls.csv; cat $f) > aaa; mv aaa $f; done
wc -l train.controls.csv.*
split -l 8 --numeric-suffixes=1 train.controls.csv train.controls.csv.
wc -l train.controls.csv.*
for f in train.controls.csv.*; do (head -n 1 train_controls.csv; cat $f) > aaa; mv aaa $f; done
wc -l train.controls.csv.*
rm -Rf new_train/*
for f in train.controls.csv.*; do python convert.py $f train new_train/ &; done
for f in train.controls.csv.*; do (python convert.py $f train new_train/ &); done
ps
screen -R c1
ls new_train/
ls train.controls.csv.* | wc -l
ls new_train/
find new_train/ -name "*.png" | wc -l
find new_train/ -name "*.png" | wc -l; date
ls new_train/*jpg | wc -l
find new_train/ -name "*.png" | wc -l; date
ls new_train/*jpg | wc -l
find new_train/ -name "*.png" | wc -l; date
top
find new_train/ -name "*.png" | wc -l; date
ls new_train/*jpg | wc -l
top
find new_train/ -name "*.png" | wc -l; date
top
ps -a | grep python | wc -l
ls new_train/*jpg | wc -l
top
ls new_train/*jpg | wc -l
find new_train/ -name "*.png" | wc -l; date
df -h .
ps -a | grep python | wc -l
find new_train/ -name "*.png" | wc -l; date
ps -a | grep python | wc -l
find new_train/ -name "*.png" | wc -l; date
ls new_train/*jpg | wc -l
ps -a | grep python | wc -l
find new_train/ -name "*.png" | wc -l; date
ps
find new_train/ -name "*.png" | wc -l; date
screen -list
screen -r c1
find new_train/ -name "*.png" | wc -l; date
ps -a | grep python | wc -l
top
find new_train/ -name "*.png" | wc -l; date
ls -ltr new_train/
ls -ltr new_train/HUVEC-05
find new_train/ -name "*.png" | wc -l; date
screen -r c1
ps -a | grep python | wc -l
find new_train/ -name "*.png" | wc -l; date
sudo mount -o discard,defaults /dev/sdb /data
cd /data/code/cells/
ls -ltr
mv new_train/ ../../input/
cd ../../input/
ls -ltr
ls new_train/
mkdir conv_imgs
mv new_train/*jpg conv_imgs/
ls
ls new_train/
cd -
vi train_negative.py 
vi ImagesDS_negative.py 
python train_negative.py new1 none my_train1N_HUVEC.set 100 > train_HUVEC_new1.out
ls ../../input/new_train/HUVEC-18/Plate3/E20_s1_w1.png
ls ../../input/new_train/HUVEC-18/
ls ../../input/new_train/
screen -R c1
screen -r c1
grep -v HUVEC-18 my_train1N_HUVEC.set > my_train1N_HUVEC2.set
vi my_train1N_HUVEC2.set
vi train_negative.py 
screen -r c1
ls -ltr
screen -r c1
top
ls -ltr | tail
cd /data/code/cells/
ls -ltr *out
vi train_HUVEC_noCOntrols_noPretrained_noNormalize.out
vi train_HUVEC_noCOntrols_noPretrained.out
vi train_HUVEC_noCOntrols_Pretrained.out 
vi train_HUVEC_noCOntrols_noPretrained_noNormalize.out
cd ../../input/
ls
mv HUVEC-18.zip new_train/
ll
ls -l
ls -lh
rm train.zip 
cd new_train/
ls
unzip HUVEC-18.zip 
ls
rm -Rf HUVEC-18.zip 
tail -f train_HUVEC_new1.out
vi ImagesDS.py 
vi ImagesDS_negative.py 
vi ImagesDS.py 
vi ImagesDS_negative.py 
python train_negative.py new1 none my_train1N_HUVEC2.set 100 > train_HUVEC_new1.out
screen -r c1
python train_negative.py tmp none my_train1N_HUVEC2.set 100
vi ImagesDS_negative.py 
python train_negative.py tmp none my_train1N_HUVEC.set 100
vi ImagesDS_negative.py 
python train_negative.py tmp none my_train1N_HUVEC.set 100
vi ImagesDS_negative.py 
python train_negative.py tmp none my_train1N_HUVEC.set 100
vi ImagesDS_negative.py 
python train_negative.py tmp none my_train1N_HUVEC.set 100
vi ImagesDS_negative.py 
vi train_negative.py 
ls
cd ../../code/cells/
python train_negative.py tmp none my_train1N_HUVEC.set 100
vi train_negative.py 
vi ImagesDS_negative.py 
git status
rm train.controls.csv*
git status
vi ImagesDS_negative.py 
ls -ltr 
python train_negative.py tmp none my_train1N_HUVEC.set 100
exit
exit
sudo mount -o discard,defaults /dev/sdb /data
cd /data/code/cells/
ls -ltr 
vi ImagesDS_negative.py 
python train_negative.py tmp none my_train1N_HUVEC.set 100
vi ImagesDS_negative.py 
python train_negative.py tmp none my_train1N_HUVEC.set 100
vi ImagesDS_negative.py 
python train_negative.py tmp none my_train1N_HUVEC.set 100
vi ImagesDS_negative.py 
python train_negative.py tmp none my_train1N_HUVEC.set 100
vi ImagesDS_negative.py 
python train_negative.py tmp none my_train1N_HUVEC.set 100
vi ImagesDS_negative.py 
python train_negative.py tmp none my_train1N_HUVEC.set 100
vi ImagesDS_negative.py 
python train_negative.py tmp none my_train1N_HUVEC.set 100
vi ImagesDS_negative.py 
python train_negative.py tmp none my_train1N_HUVEC.set 100
vi ImagesDS_negative.py 
python train_negative.py tmp none my_train1N_HUVEC.set 100
vi ImagesDS_negative.py 
python train_negative.py tmp none my_train1N_HUVEC.set 100
vi ImagesDS_negative.py 
vi train_negative.py 
vi ImagesDS_negative.py 
cd /data/code/cells/
vi train_negative.py 
top
ls -ltr
git status
git diff train_negative.py | less
git diff ImagesDS_negative.py | less
python train_negative.py tmp none my_train1N_HUVEC.set 100
vi ImagesDS_negative.py 
top
vi train_negative.py 
vi ImagesDS_negative.py 
python train_negative.py tmp none my_train1N_HUVEC.set 100
vi train_negative.py 
vi ImagesDS_negative.py 
vi train_negative.py 
vi ImagesDS_negative.py 
vi train_negative.py 
ls -ltr | tail
python train_negative.py tmpCROP best_model_regular_losstmp.bin my_train1N_HUVEC.set 100
vi train_negative.py 
vi ImagesDS_negative.py 
python train_negative.py tmpCROP2 best_model_regular_losstmpCROP.bin my_train1N_HUVEC.set 100
vi train_negative.py 
vi ImagesDS_negative.py 
python train_negative.py tmpCROP3 best_model_regular_losstmpCROP2.bin my_train1N_HUVEC.set 100
vi train_negative.py 
vi ImagesDS_negative.py 
vi train_negative.py 
vi ImagesDS_negative.py 
vi train_negative.py 
python train_negative.py tmpCROP2 best_model_regular_losstmpCROP2.bin my_train1N_HUVEC.set 100
exit
exit
sudo mount -o discard,defaults /dev/sdb /data
cd /data/code/cells/
ls -ltr
cd /data/code/cells/
python train_negative.py tmpCROP2 best_model_regular_losstmpCROP2.bin my_train1N_HUVEC.set 100
vi ImagesDS_negative.py 
vi train_negative.py 
vi ImagesDS_negative.py 
vi train_negative.py 
vi ImagesDS_negative.py 
ls -ltr ImagesDS_*
ls -ltr ImagesDS*
vi ImagesDS.py 
git status
git add *py
git status
git commit -a
git pull
git push
vi ImagesDS.py 
python train_negative.py tmpO none my_train1N_HUVEC.set 100
vi train_regular.py 
sudo mount -o discard,defaults /dev/sdb /data
cd /data/code/cells/
vi train_regular.py 
clear
sudo mount -o discard,defaults /dev/sdb /data
cd /data/code/cells/
cd /data/code/cells/
vi train_regular.py 
rm .train_regular.py.sw
rm .train_regular.py.sw*
vi train_regular.py 
git commit -a
git push
vi train_regular.py 
vi ImagesDS.py 
sudo mount -o discard,defaults /dev/sdb /data
cd /data/code/cells/
vi ImagesDS.py 
rm .ImagesDS.py.swp 
vi ImagesDS.py 
vi train_regular.py 
vi ImagesDS.py 
vi DensNet.py 
ls -ltr
ls -ltr ../../input/wc -l my_train*set
wc -l ../../input/wc -l my_train*set
grep HUVEC-18 my_train1B.set | wc -l
vi ~/.bash_history 
ls -ltr
python train_regular.py tmp1 none my_extended_train.set 100
vi train_regular.py 
vi train_regular.py 
python train_regular.py tmp1 none my_train1N_HUVEC.set 100
vi DensNet.py 
wc -l my_train1N_HUVEC.set
wc -l my_extended_train.set 
vi train_regular.py 
ls *bin | wc -l
ls *bin | less
rm *bin
ls
vi my_extended_train.set 
grep HUVEC-18 my_extended_train.set > aaa; cat my_extended_train.set aaa > my_extended_train2.set
vi my_extended_train2.set
wc -l my_extended_train.set my_extended_train2.set
python train_regular.py all1 none my_extended_train.set 100
clear
ls -ltr | tail
vi train_regular.py 
top
exit
screen -R c1
exit
screen -R c1
pip install efficientnet_pytorch
pip install --user efficientnet_pytorch
cd /data/code/cells/
screen -r c1
exit
screen -r c1
exit
python train_regular.py all1 none my_extended_train2.set 100
screen -r c1
cd /data/code/cells/
ls -ltr
vi train_regular.py 
vi DensNet.py 
vi ImagesDS.py 
screen -r c1
vi ImagesDS.py 
vi train_regular.py 
screen -r c1
exit
python train_regular.py single1 final_model_all1.bin my_extended_train2.set 100
screen -r c1
sudo mount -o discard,defaults /dev/sdb /data
cd /data/code/cells/
ls -ltr
screen -R c1
ls -ltr
vi train_regular.py 
python train_regular.py single2 final_model_single1.bin my_extended_train2.set 100
screen -R c1
vi train_regular.py 
screen -R c1
exit
screen -R c1
exit
screen -r c1
cd /data/code/cells/
ls -ltr | tail
screen -r c1
ls -ltr | tail
screen -r c1
ls -ltr | tail
screen -r c1
vi DensNet.py 
python train_regular.py s264_1 none my_extended_train2.set 100
screen -r c1
vi DensNet.py 
python train_regular.py s201_1 none my_extended_train2.set 100
screen -r c1
vi train_regular.py 
screen -r c1
vi train_regular.py 
screen -r c1
vi DensNet.py 
screen -r c1
vi train_regular.py 
screen -r c1
vi train_regular.py 
python train_regular.py s264_1 none my_extended_train2.set 100
screen -r c1
vi train_regular.py 
screen -r c1
cd ..
cd ../input/
ls
less test.folders 
ls
top
zip -r train.zip train
ls -ltr
man tar
screen -r c1
cd /data/input/
ls -ltr
ls -ltrh
du --max-depth=1 -h .
ls -ltrh
ls -ltrh train.tar 
df -h .
ls -ltrh train.tar 
tar -c train > train.tar
ls /home/
ls /home/pmajek/
exit
sftp 35.204.173.55
sftp 10.164.0.2
cd ~
ls
mkdir .ssh
ls .ssh/
mv id_rsa* .ssh/
sftp 35.204.173.55
screen -r c1
cd /data/code/cells/
git sstatus
git status
git commit -a
git push
screen -r c1
vi ~/.vimrc 
sftp 35.204.173.55
vi ~/.bash_history 
cd /data/code/cells/
ls -ltr *bin
sftp 35.204.173.55
ls -ltr *sirna
ls -ltr *sirnas
ls
sftp 35.204.173.55
vi ~/.bash_history 
sftp 35.204.173.55
vi ~/.bashrc 
top
python train_regular.py s201_1 none my_extended_train2.set 100
sudo .mount -o discard,defaults /dev/sdb /data
sudo mount -o discard,defaults /dev/sdb /data
cd /data/code/cells/
vi train_regular.py 
ls -ltr *bin
cd /data/code/cells/
sftp 35.204.173.55
ls -ltr | tail
python train_regular.py s201_2 best_model_regular_losss201_1.bin my_extended_train2.set 100
ls -ltr | tail
vi train_regular.py 
screen -list
screen -r c1
screen -R c1
ls -ltr | tail
tail -f single201_3.log 
top
exit
exit
cd /data/code/cells/
tail -f single201_3.log 
vi single201_3.log 
cd /data/code/cells/
vi train_controls.py 
less my_train1C.set 
less my_train2C.set 
vi ~/.bash_history 
ls cc.1
ls cc/
less cc/cc.1
less cc/cc.6 
ls | grep cc
vi cc
vi ~/.bash_history 
less features_test1_1.csv
ls features_*
vi ~/.bash_history 
ls *features*py
vi features_cosFace.py 
vi ~/.bash_history 
ls best_model_cosFace_loss2Db.bin
tail -f single201_3.log 
sudo mount -o discard,defaults /dev/sdb /data
cd /data/code/cells/
vi train_regular.py 
ls -ltr | tail
screen -R c1
ls -ltr | tail
tail -f single201_4.log
exit
cd /data/code/cells/
ls -ltr | tail
tail -f single201_4.log
ls -ltr *bin
git status
git diff train_regular.py 
vi train_regular.py 
git diff train_regular.py 
git status
git pull
vi single201_4.log
python train_regular.py s201_4 final_model_s201_3.bin my_extended_train2.set 100 > single201_4.log 
ll best_model_regular_losss201_4.bin
ls -l best_model_regular_losss201_4.bin
ls -ltr *out
ls -ltr *log
vi single201_4.log
time python predict_regular.py CC1 best_model_regular_losss201_4.bin test1.set sirnas.1 submission1.csv 201 2;
head submission1.csv 
mv submission1.csv submission201_1.csv
sftp 35.204.173.55
time python predict_regular.py CC1 best_model_regular_losss201_4.bin test2.set sirnas.2 submission201_2.csv 121 2;  time python predict_regular.py CC1 best_model_regular_losss201_4.bin test3.set sirnas.3 submission201_3.csv 121 2;   time python predict_regular.py CC1 best_model_regular_losss201_4.bin test4.set sirnas.4 submission201_4.csv 121 2;
time python predict_regular.py CC1 best_model_regular_losss201_4.bin test2.set sirnas.2 submission201_2.csv 201 2;  time python predict_regular.py CC1 best_model_regular_losss201_4.bin test3.set sirnas.3 submission201_3.csv 201 2;   time python predict_regular.py CC1 best_model_regular_losss201_4.bin test4.set sirnas.4 submission201_4.csv 201 2;
sftp 35.204.173.55
ls -ltr | tail
rm test\?.set 
sftp 35.204.173.55
time python predict_negative.py CC1 best_model_regular_lossnegative1.bin test1.set sirnas.1 submission_neg1.csv 121 2;
vi predict_negative.py 
sftp 35.204.173.55
time python predict_negative.py CC1 best_model_regular_lossnegative1.bin test1.set sirnas.1 submission_neg1.csv 121 2;
sftp 35.204.173.55
for in in 1 2 3 4; do time python predict_negative.py CC1 best_model_regular_lossnegative1.bin test${i}.set sirnas.$i submission_neg${i}.csv 121 2; done
for i in in 1 2 3 4; do time python predict_negative.py CC1 best_model_regular_lossnegative1.bin test${i}.set sirnas.$i submission_neg${i}.csv 121 2; done
clear
for i in in 1 2 3 4; do time python predict_negative.py CC1 best_model_regular_lossnegative1.bin test${i}.set sirnas.$i submission_neg${i}.csv 121 2; done
for i in 1 2; do time python predict_negative.py CC1 best_model_regular_lossnegative1.bin test${i}.set sirnas.$i submission_neg${i}.csv 121 2; done
ls -ltr | tail
sftp 35.204.173.55
ls -ltr | tail
head submission201_1.csv 
for i in 1 2; do time python predict_regular.py CC1 best_model_regular_losss201_4.bin test${i}.set sirnas.${i} submission201_${i}.csv 201 9; done
ls -ltr | tail
sftp 35.204.173.55
for i in 4; do time python predict_regular.py CC1 best_model_regular_losss201_4.bin test${i}.set sirnas.${i} submission201_${i}.csv 201 9; done
exit
exit
screen -r c1
exit
sudo mount -o discard,defaults /dev/sdb /data
cd /data/code/cells/
