
STEPSIZE=50000

mkdir output/filterwidth

for FILTERWIDTH in 1 2 4 8 16 32 64 128 256
do
    for ITERATION in {1..10}
    do
        python trainer/task.py --train-files ./data/filtered/output.* --job-dir ./output/filterwidth/$FILTERWIDTH --sample_size $FILTERWIDTH --train-steps $(($ITERATION * $STEPSIZE))
        python generate.py --samples 1000 --out_path ./output/filterwidth/notalk\_$FILTERWIDTH\_$(($ITERATION * $STEPSIZE)) --dat_seed ./data/filtered/output.dat output/filterwidth/$FILTERWIDTH/model.ckpt-$(($ITERATION * $STEPSIZE))
        mkdir -p output/filterwidth/backup/$FILTERWIDTH
        cp output/filterwidth/$FILTERWIDTH/model.ckpt-$(($ITERATION * $STEPSIZE))* output/filterwidth/backup/$FILTERWIDTH/
    done
done

