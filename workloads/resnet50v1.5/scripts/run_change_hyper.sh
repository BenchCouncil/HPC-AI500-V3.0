echo "start wd 5e05 ..."
sleep 10

./fp32_change_hyper_8_8_wd5e05.sh

echo "start momentum ..."
sleep 60

./fp32_change_hyper_momentum_8_8.sh

sleep 60
./fp32_change_hyper_momentum_4_4.sh
