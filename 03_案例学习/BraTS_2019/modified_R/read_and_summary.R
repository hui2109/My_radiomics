r_summary <- function(csv_path, encoding){
  csv_table <- read.csv(csv_path, fileEncoding = encoding)
  csv_table <- data.frame(csv_table)
  # 打印临床特征名
  mnames <- names(csv_table)
  # 查看临床特征的基本信息
  msummary <- summary(csv_table)
  return(csv_table, mnames, msummary)
}