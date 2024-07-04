use anyhow::Result;
use linfa_clustering::KMeans;
use polars::prelude::*;
use reqwest;
use std::io::Write;
use tempfile::tempfile;
use tempfile::NamedTempFile;

struct DataFrameSummary(DataFrame);

impl DataFrameSummary {
    fn describe(&self) -> Result<DataFrame> {
        let df = &self.0;
        let df_mean = DataFrame::new(
            df.iter()
                .map(|s| s.mean_reduce().into_series(s.name()))
                .collect::<Vec<_>>(),
        )?;
        dbg!(&df_mean);
        let df_min = DataFrame::new(
            df.iter()
                .map(|s| {
                    s.min_reduce()
                        .expect("data frame should not be empty")
                        .into_series(s.name())
                })
                .collect::<Vec<_>>(),
        )?;
        dbg!(&df_mean);
        let df_q1 = DataFrame::new(
            df.iter()
                .map(|s| {
                    s.quantile_reduce(0.25, QuantileInterpolOptions::Linear)
                        .expect("data frame should not be empty")
                        .into_series(s.name())
                })
                .collect::<Vec<_>>(),
        )?;
        dbg!(&df_q1);
        let df_median = DataFrame::new(
            df.iter()
                .map(|s| {
                    s.median_reduce()
                        .expect("data frame should not be empty")
                        .into_series(s.name())
                })
                .collect::<Vec<_>>(),
        )?;
        dbg!(&df_median);
        let df_q3 = DataFrame::new(
            df.iter()
                .map(|s| {
                    s.quantile_reduce(0.75, QuantileInterpolOptions::Linear)
                        .expect("data frame should not be empty")
                        .into_series(s.name())
                })
                .collect::<Vec<_>>(),
        )?;
        dbg!(&df_q3);
        let df_max = DataFrame::new(
            df.iter()
                .map(|s| {
                    s.max_reduce()
                        .expect("data frame should not be empty")
                        .into_series(s.name())
                })
                .collect::<Vec<_>>(),
        )?;
        dbg!(&df_max);

        let result = concat(
            [
                df_min.lazy(),
                df_q1.lazy(),
                df_median.lazy(),
                df_q3.lazy(),
                df_max.lazy(),
            ],
            UnionArgs::default(),
        )?
        .collect()?;
        dbg!(&result);
        let labels = df!(""=>["min", "q1", "med", "q3", "max"])?;
        dbg!(&labels);
        Ok(polars::functions::concat_df_horizontal(&[labels, result])?)
    }
}

fn main() -> Result<()> {
    // get dataset
    let data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data";
    let body = reqwest::blocking::get(data_url)?.text()?;

    // write dataset to file
    let mut data_file = tempfile()?;
    write!(data_file, "{}", &body)?;

    let temp_file = NamedTempFile::new()?;
    write!(&temp_file, "{}", &body)?;
    dbg!(&data_file);
    dbg!(&temp_file);
    dbg!(temp_file.into_temp_path());

    // read CSV file
    let lf = CsvReadOptions::default()
        .with_has_header(false)
        .into_reader_with_file_handle(data_file);
    // let lf = CsvReader::new(data_file);
    let mut df: DataFrame = lf.finish()?;
    let columns = vec![
        "class",
        "Alcohol",
        "Malic acid",
        "Ash",
        "Alcalinity of ash",
        "Magnesium",
        "Total phenols",
        "Flavanoids",
        "Nonflavanoid phenols",
        "Proanthocyanins",
        "Color intensity",
        "Hue",
        "OD280/OD315 of diluted wines",
        "Proline",
    ];
    let _ = df.set_column_names(&columns)?;
    dbg!(&df);
    dbg!(&df.iter().map(|s| s.min::<f64>()).collect::<Vec<_>>());

    // rebuild description
    let descr = DataFrameSummary(df.clone()).describe()?;
    dbg!(descr);

    Ok(())
}
