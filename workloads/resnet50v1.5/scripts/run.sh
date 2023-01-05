echo "start fp16-profiling ..."
sleep 10

./fp16.sh

echo "start fp32-profiling ..."
sleep 60

./fp32.sh
