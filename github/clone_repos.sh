function clone_em_all (){
  mkdir -p data/ subcmds/
  grep "\"url\":" $1 | uniq | sed "s/^.*\"url\": /git clone /" | \
      sed  "s/github\.com\/\(.*\)\/\(.*\)\"/github.com\/\1\/\2\"  \.\.\/data\/\1__-__\2/" > subcmds_master
  let "b=$(wc -l < subcmds_master)/4"
  split -l $b subcmds_master subcmds/sub_bash_
  
}

clone_em_all $1
