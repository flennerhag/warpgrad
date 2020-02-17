set -e

cd src

while getopts ":pdf" opt; do
  case $opt in
    p)
      echo "installing packages"

      cd warpgrad 
      pip install -e .

      cd ../leap
      pip install -e .

      cd ../maml
      pip install -e .
      
      cd ..
      echo "done"     
      ;;
    d)
      echo "Getting data"
      
      cd omniglot
      bash getdata.sh
      
      cd ..
      echo "done"
      ;;
    l)
      echo "making log dirs"
      cd omniglot

      mkdir -p logs
      mkdir -p logs/warp_leap
      mkdir -p logs/leap
      mkdir -p logs/reptile
      mkdir -p logs/maml
      mkdir -p logs/fomaml
      mkdir -p logs/ft
      mkdir -p logs/no
      echo "done"

      cd ..
      cd ..

      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      ;;
  esac
done
