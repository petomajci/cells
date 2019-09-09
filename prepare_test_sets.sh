echo id_code,experiment,plate,well > test1.set
echo id_code,experiment,plate,well > test2.set
echo id_code,experiment,plate,well > test3.set
echo id_code,experiment,plate,well > test4.set

grep -P "HEPG2-08|HUVEC-21|HUVEC-24|U2OS-05" ../input/test.csv | grep ",1," >> test2.set
grep -P "HEPG2-08|HUVEC-21|HUVEC-24|U2OS-05" ../input/test.csv | grep ",2," >> test3.set
grep -P "HEPG2-08|HUVEC-21|HUVEC-24|U2OS-05" ../input/test.csv | grep ",3," >> test4.set
grep -P "HEPG2-08|HUVEC-21|HUVEC-24|U2OS-05" ../input/test.csv | grep ",4," >> test1.set

grep -P "HEPG2-09|RPE-08" ../input/test.csv | grep ",1," >> test1.set
grep -P "HEPG2-09|RPE-08" ../input/test.csv | grep ",2," >> test2.set
grep -P "HEPG2-09|RPE-08" ../input/test.csv | grep ",3," >> test3.set
grep -P "HEPG2-09|RPE-08" ../input/test.csv | grep ",4," >> test4.set

grep -P "HEPG2-10|HEPG2-11|HUVEC-17|HUVEC-18|HUVEC-22|HUVEC-23|RPE-09|RPE-10|RPE-11" ../input/test.csv | grep ",1," >> test3.set
grep -P "HEPG2-10|HEPG2-11|HUVEC-17|HUVEC-18|HUVEC-22|HUVEC-23|RPE-09|RPE-10|RPE-11" ../input/test.csv | grep ",2," >> test4.set
grep -P "HEPG2-10|HEPG2-11|HUVEC-17|HUVEC-18|HUVEC-22|HUVEC-23|RPE-09|RPE-10|RPE-11" ../input/test.csv | grep ",3," >> test1.set
grep -P "HEPG2-10|HEPG2-11|HUVEC-17|HUVEC-18|HUVEC-22|HUVEC-23|RPE-09|RPE-10|RPE-11" ../input/test.csv | grep ",4," >> test2.set

grep -P "HUVEC-19|HUVEC-20|U2OS-04" ../input/test.csv | grep ",1," >> test4.set
grep -P "HUVEC-19|HUVEC-20|U2OS-04" ../input/test.csv | grep ",2," >> test1.set
grep -P "HUVEC-19|HUVEC-20|U2OS-04" ../input/test.csv | grep ",3," >> test2.set
grep -P "HUVEC-19|HUVEC-20|U2OS-04" ../input/test.csv | grep ",4," >> test3.set
