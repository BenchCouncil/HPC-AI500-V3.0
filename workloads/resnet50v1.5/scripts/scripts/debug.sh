
cat metrics | while read metric_name iter_step
do
    echo ${metric_name}": "${iter_step}
done
