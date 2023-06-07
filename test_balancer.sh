# cd /efs/hang/telecombrain/globecom23/BERT_FT_3GPP


## COPY:
#find /efs/hang/telecombrain/globecom23/3GPP_SOURCES/RAN2 -maxdepth 1 -type f -name 'R2-2*' \
#-exec cp -i {} -t /efs/hang/telecombrain/globecom23/3GPP_SOURCES/3G/NEW/RAN2  \;

find /efs/hang/telecombrain/globecom23/Paragraphs/5G/NEW/CT1 -maxdepth 1 -type f -name 'C1-2[0]*' \
-exec cp -i {} -t /efs/hang/telecombrain/globecom23/Paragraphs/5G/OLD_ONE/CT1 \;
find /efs/hang/telecombrain/globecom23/Paragraphs/5G/NEW/CT3 -maxdepth 1 -type f -name 'C3-2[0]*' \
-exec cp -i {} -t /efs/hang/telecombrain/globecom23/Paragraphs/5G/OLD_ONE/CT3 \;
find /efs/hang/telecombrain/globecom23/Paragraphs/5G/NEW/CT4 -maxdepth 1 -type f -name 'C4-2[0]*' \
-exec cp -i {} -t /efs/hang/telecombrain/globecom23/Paragraphs/5G/OLD_ONE/CT4 \;
find /efs/hang/telecombrain/globecom23/Paragraphs/5G/NEW/CT6 -maxdepth 1 -type f -name 'C6-2[0]*' \
-exec cp -i {} -t /efs/hang/telecombrain/globecom23/Paragraphs/5G/OLD_ONE/CT6 \;


find /efs/hang/telecombrain/globecom23/Paragraphs/5G/NEW/RAN1 -maxdepth 1 -type f -name 'R1-2[0]*' \
-exec cp -i {} -t /efs/hang/telecombrain/globecom23/Paragraphs/5G/OLD_ONE/RAN1 \;
find /efs/hang/telecombrain/globecom23/Paragraphs/5G/NEW/RAN2 -maxdepth 1 -type f -name 'R2-2[0]*' \
-exec cp -i {} -t /efs/hang/telecombrain/globecom23/Paragraphs/5G/OLD_ONE/RAN2 \;
find /efs/hang/telecombrain/globecom23/Paragraphs/5G/NEW/RAN3 -maxdepth 1 -type f -name 'R3-2[0]*' \
-exec cp -i {} -t /efs/hang/telecombrain/globecom23/Paragraphs/5G/OLD_ONE/RAN3 \;
find /efs/hang/telecombrain/globecom23/Paragraphs/5G/NEW/RAN4 -maxdepth 1 -type f -name 'R4-2[0]*' \
-exec cp -i {} -t /efs/hang/telecombrain/globecom23/Paragraphs/5G/OLD_ONE/RAN4 \;
find /efs/hang/telecombrain/globecom23/Paragraphs/5G/NEW/RAN5 -maxdepth 1 -type f -name 'R5-2[0]*' \
-exec cp -i {} -t /efs/hang/telecombrain/globecom23/Paragraphs/5G/OLD_ONE/RAN5 \;

find /efs/hang/telecombrain/globecom23/Paragraphs/5G/NEW/SA1 -maxdepth 1 -type f -name 'S1-2[0]*' \
-exec cp -i {} -t /efs/hang/telecombrain/globecom23/Paragraphs/5G/OLD_ONE/SA1 \;
find /efs/hang/telecombrain/globecom23/Paragraphs/5G/NEW/SA2 -maxdepth 1 -type f -name 'S2-2[0]*' \
-exec cp -i {} -t /efs/hang/telecombrain/globecom23/Paragraphs/5G/OLD_ONE/SA2 \;
find /efs/hang/telecombrain/globecom23/Paragraphs/5G/NEW/SA3 -maxdepth 1 -type f -name 'S3-2[0]*' \
-exec cp -i {} -t /efs/hang/telecombrain/globecom23/Paragraphs/5G/OLD_ONE/SA3 \;
find /efs/hang/telecombrain/globecom23/Paragraphs/5G/NEW/SA4 -maxdepth 1 -type f -name 'S4-2[0]*' \
-exec cp -i {} -t /efs/hang/telecombrain/globecom23/Paragraphs/5G/OLD_ONE/SA4 \;
find /efs/hang/telecombrain/globecom23/Paragraphs/5G/NEW/SA5 -maxdepth 1 -type f -name 'S5-2[0]*' \
-exec cp -i {} -t /efs/hang/telecombrain/globecom23/Paragraphs/5G/OLD_ONE/SA5 \;
find /efs/hang/telecombrain/globecom23/Paragraphs/5G/NEW/SA6 -maxdepth 1 -type f -name 'S6-2[0]*' \
-exec cp -i {} -t /efs/hang/telecombrain/globecom23/Paragraphs/5G/OLD_ONE/SA6 \;

## FIND:
#find /efs/hang/telecombrain/globecom23/3GPP_SOURCES/4G/OLD/CT1 -maxdepth 1 -type f -name '*' | wc -l
#find /efs/hang/telecombrain/globecom23/3GPP_SOURCES/4G/OLD/CT3 -maxdepth 1 -type f -name '*' | wc -l
#find /efs/hang/telecombrain/globecom23/3GPP_SOURCES/4G/OLD/CT4 -maxdepth 1 -type f -name '*' | wc -l
#find /efs/hang/telecombrain/globecom23/3GPP_SOURCES/4G/OLD/CT6 -maxdepth 1 -type f -name '*' | wc -l
#find /efs/hang/telecombrain/globecom23/3GPP_SOURCES/4G/OLD/SA5 -maxdepth 1 -type f -name '*' | wc -l
#find /efs/hang/telecombrain/globecom23/3GPP_SOURCES/4G/OLD/SA6 -maxdepth 1 -type f -name '*' | wc -l

echo
#find /efs/hang/telecombrain/globecom23/3GPP_SOURCES/4G/OLD/CT1 -maxdepth 1 -type f -name 'C1-1[5-9]*' | wc -l
#find /efs/hang/telecombrain/globecom23/3GPP_SOURCES/4G/OLD/CT3 -maxdepth 1 -type f -name 'C3-1[5-9]*' | wc -l
#find /efs/hang/telecombrain/globecom23/3GPP_SOURCES/4G/OLD/CT4 -maxdepth 1 -type f -name 'C4-1[5-9]*' | wc -l
#find /efs/hang/telecombrain/globecom23/3GPP_SOURCES/4G/OLD/CT6 -maxdepth 1 -type f -name 'C6-1[5-9]*' | wc -l
#find /efs/hang/telecombrain/globecom23/3GPP_SOURCES/4G/OLD/SA5 -maxdepth 1 -type f -name 'S5-1[5-9]*' | wc -l
#find /efs/hang/telecombrain/globecom23/3GPP_SOURCES/4G/OLD/SA6 -maxdepth 1 -type f -name 'S6-1[5-9]*' | wc -l

echo

#find /efs/hang/telecombrain/globecom23/3GPP_SOURCES/4G/NEW/CT1 -maxdepth 1 -type f -name '*' | wc -l
#find /efs/hang/telecombrain/globecom23/3GPP_SOURCES/4G/NEW/CT3 -maxdepth 1 -type f -name '*' | wc -l
#find /efs/hang/telecombrain/globecom23/3GPP_SOURCES/4G/NEW/CT4 -maxdepth 1 -type f -name '*' | wc -l
#find /efs/hang/telecombrain/globecom23/3GPP_SOURCES/4G/NEW/CT6 -maxdepth 1 -type f -name '*' | wc -l
#find /efs/hang/telecombrain/globecom23/3GPP_SOURCES/4G/NEW/SA5 -maxdepth 1 -type f -name '*' | wc -l
#find /efs/hang/telecombrain/globecom23/3GPP_SOURCES/4G/NEW/SA6 -maxdepth 1 -type f -name '*' | wc -l
#echo

#find /efs/hang/telecombrain/globecom23/Paragraphs/3G/NEW/RAN1 -maxdepth 1 -type f -name '*.zip*' | wc -l
#find /efs/hang/telecombrain/globecom23/Paragraphs/3G/NEW/RAN2 -maxdepth 1 -type f -name '*.zip*' | wc -l
#find /efs/hang/telecombrain/globecom23/Paragraphs/3G/NEW/RAN3 -maxdepth 1 -type f -name '*.zip*' | wc -l
#find /efs/hang/telecombrain/globecom23/Paragraphs/3G/NEW/RAN4 -maxdepth 1 -type f -name '*.zip*' | wc -l
#find /efs/hang/telecombrain/globecom23/Paragraphs/3G/NEW/RAN5 -maxdepth 1 -type f -name '*.zip*' | wc -l
#echo
#find /efs/hang/telecombrain/globecom23/Paragraphs/4G/OLD/RAN1 -maxdepth 1 -type f -name '*.zip*' | wc -l
#find /efs/hang/telecombrain/globecom23/Paragraphs/4G/OLD/RAN2 -maxdepth 1 -type f -name '*.zip*' | wc -l
#find /efs/hang/telecombrain/globecom23/Paragraphs/4G/OLD/RAN3 -maxdepth 1 -type f -name '*.zip*' | wc -l
#find /efs/hang/telecombrain/globecom23/Paragraphs/4G/OLD/RAN4 -maxdepth 1 -type f -name '*.zip*' | wc -l
#find /efs/hang/telecombrain/globecom23/Paragraphs/4G/OLD/RAN5 -maxdepth 1 -type f -name '*.zip*' | wc -l

## MOVING: 
#find /efs/hang/telecombrain/globecom23/3GPP_SOURCES/3G/OLD/RAN1 -maxdepth 1 -type f -name '*XXX*' \
#-exec mv {} -t /efs/hang/telecombrain/globecom23/3GPP_Others/RAN1 \;
#mv /efs/hang/telecombrain/globecom23/3GPP_SOURCES/CT1/CT3/* /efs/hang/telecombrain/globecom23/3GPP_SOURCES/FULL/CT3 


#input_path="/efs/hang/telecombrain/globecom23/Paragraphs/5G/NEW"
#output_path="/efs/hang/telecombrain/globecom23/Paragraphs/5G_LATEST/NEW"

#for folder in CT1 CT3 CT4 CT6 RAN1 RAN2 RAN3 RAN4 RAN5 SA1 SA2 SA3 SA4 SA5 SA6; do
  #find ${input_path}/${folder} -type f -exec cp {} ${output_path}/${folder}/ \;
#done


## DELETE:
#find /efs/hang/telecombrain/globecom23/3GPP_SOURCES/3G/OLD/CT1 -maxdepth 1 -type f -name '*rev*' \
#-exec rm -i {};


