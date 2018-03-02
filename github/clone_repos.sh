function line_em_up (){
  mkdir -p data/ subcmds/
  grep "\"url\":" $1 | uniq | sed "s/^.*\"url\": /git clone --depth=2 --shallow-submodules /" | \
      sed  "s/github\.com\/\(.*\)\/\(.*\)\"/github\.com\/\1\/\2\"  \.\/data\/\1__-__\2  \&\& \\\\/" > subcmds_master

  let "b=$(wc -l < subcmds_master)/$2"
  split -l $b subcmds_master subcmds/sub_bash_
  chmod 755 subcmds/sub_bash_*
}

function clone_em_all (){
  for x in $(ls subcmds); do
    echo "Executing $x"
    echo "echo 'done'" >> subcmds/$x
    ./subcmds/$x &
  done
}

echo "generating $2 bash cmds"
line_em_up $1 $2

read -p "Do you want to clone $(wc -l <subcmds_master) repos? [Yy]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]];then
  clone_em_all
else
  echo "Aborting"
fi

