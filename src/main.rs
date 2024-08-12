use anyhow::Result;
use itertools::Itertools;
use linfa::prelude::Predict; // to use Kmeas.predict()
use linfa::traits::Fit;
use linfa::DatasetBase;
use linfa_clustering::KMeans;
use linfa_nn::distance::L2Dist;
use polars::prelude::*;
use std::collections::HashMap;
use std::collections::HashSet;
use std::io::Write;
use std::iter::zip;
use tempfile::tempfile;
use tempfile::NamedTempFile;

#[derive(Debug, Copy, Clone)]
enum Normalization {
    /// Bring standard deviation to 1 and average to zero
    Standard,
    /// Bring values in [0, 1]
    MinMax,
    /// Bring q1 to -1 and q3 to 1
    Quartiles,
    /// Bring average to zero
    Center,
}

struct SeriePreproc(Series);

impl SeriePreproc {
    fn normalize(&self, kind: Normalization) -> Result<Series> {
        match kind {
            Normalization::Standard => Ok(self.standard()?.with_name(self.0.name())),
            Normalization::MinMax => Ok(self.minmax()?.with_name(self.0.name())),
            Normalization::Quartiles => Ok(self.quartiles()?.with_name(self.0.name())),
            Normalization::Center => Ok(self.center()?.with_name(self.0.name())),
        }
    }

    fn standard(&self) -> Result<Series> {
        let s = &self.0.cast(&DataType::Float64)?;
        let mean = s.mean_reduce().as_any_value().extract::<f64>().unwrap();
        let std = s.std(1).expect("Serie should not be empty");
        Ok(s.iter()
            .map(|elt| (elt.extract::<f64>().unwrap() - mean) / std)
            .collect())
    }

    fn minmax(&self) -> Result<Series> {
        let s = &self.0.cast(&DataType::Float64)?;
        let min = s.min_reduce()?.as_any_value().extract::<f64>().unwrap();
        let max = s.max_reduce()?.as_any_value().extract::<f64>().unwrap();
        Ok(s.iter()
            .map(|elt| (elt.extract::<f64>().unwrap() - min) / (max - min))
            .collect())
    }
    fn quartiles(&self) -> Result<Series> {
        let s = &self.0.cast(&DataType::Float64)?;
        let q1 = s
            .quantile_reduce(0.25, QuantileInterpolOptions::Linear)?
            .as_any_value()
            .extract::<f64>()
            .unwrap();
        let q3 = s
            .quantile_reduce(0.75, QuantileInterpolOptions::Linear)?
            .as_any_value()
            .extract::<f64>()
            .unwrap();
        let center = (q1 + q3) / 2_f64;
        let inter_q = q3 - q1;
        Ok(s.iter()
            .map(|elt| (elt.extract::<f64>().unwrap() - center) / inter_q)
            .collect())
    }
    fn center(&self) -> Result<Series> {
        let s = &self.0.cast(&DataType::Float64)?;
        let mean = s.mean_reduce().as_any_value().extract::<f64>().unwrap();
        Ok(s.iter()
            .map(|elt| elt.extract::<f64>().unwrap() - mean)
            .collect())
    }
}

struct DataFramePreproc(DataFrame);

impl DataFramePreproc {
    fn describe(&self) -> Result<DataFrame> {
        let df = &self.0;
        let df_min = DataFrame::new(
            df.iter()
                .map(|s| {
                    s.min_reduce()
                        .expect("data frame should not be empty")
                        .into_series(s.name())
                        .cast(&DataType::Float64) // Cast to avoid mixing i64 and f64
                        .expect("cast from int to float should go smoothly")
                })
                .collect::<Vec<_>>(),
        )?;
        let df_q1 = DataFrame::new(
            df.iter()
                .map(|s| {
                    s.quantile_reduce(0.25, QuantileInterpolOptions::Linear)
                        .expect("data frame should not be empty")
                        .into_series(s.name())
                })
                .collect::<Vec<_>>(),
        )?;
        let df_median = DataFrame::new(
            df.iter()
                .map(|s| {
                    s.median_reduce()
                        .expect("data frame should not be empty")
                        .into_series(s.name())
                        .cast(&DataType::Float64)
                        .expect("cast from int to float should go smoothly")
                })
                .collect::<Vec<_>>(),
        )?;
        let df_q3 = DataFrame::new(
            df.iter()
                .map(|s| {
                    s.quantile_reduce(0.75, QuantileInterpolOptions::Linear)
                        .expect("data frame should not be empty")
                        .into_series(s.name())
                })
                .collect::<Vec<_>>(),
        )?;
        let df_max = DataFrame::new(
            df.iter()
                .map(|s| {
                    s.max_reduce()
                        .expect("data frame should not be empty")
                        .into_series(s.name())
                        .cast(&DataType::Float64)
                        .expect("cast from int to float should go smoothly")
                })
                .collect::<Vec<_>>(),
        )?;

        let df_std = DataFrame::new(
            // ddof argument of std documented at
            // https://github.com/pola-rs/polars/blob/daf2e4983b6d94b06f2eaa3a77c2e02c112f5675/py-polars/polars/expr/list.py#L300
            df.iter()
                .map(|s| {
                    s.std_reduce(1)
                        .expect("data frame should not be empty")
                        .into_series(s.name())
                        .cast(&DataType::Float64)
                        .expect("cast from int to float should go smoothly")
                })
                .collect::<Vec<_>>(),
        )?;

        let df_mean = DataFrame::new(
            df.iter()
                .map(|s| {
                    s.mean_reduce()
                        .into_series(s.name())
                        .cast(&DataType::Float64)
                        .expect("cast from int to float should go smoothly")
                })
                .collect::<Vec<_>>(),
        )?;

        let result = concat(
            [
                df_min.lazy(),
                df_q1.lazy(),
                df_median.lazy(),
                df_q3.lazy(),
                df_max.lazy(),
                df_mean.lazy(),
                df_std.lazy(),
            ],
            UnionArgs::default(),
        )?
        .collect()?;
        let labels = df!(""=>["min", "q1", "med", "q3", "max", "avg", "std dev"])?;
        Ok(polars::functions::concat_df_horizontal(&[labels, result])?)
    }

    fn normalize(&self, kind: Normalization) -> Result<DataFrame> {
        let df = DataFrame::new(
            self.0
                .iter()
                .map(|s| SeriePreproc(s.clone()).normalize(kind).unwrap())
                .collect::<Vec<_>>(),
        )?;
        Ok(df)
    }
}

fn swap_series(s: Series, swap: HashMap<AnyValue, AnyValue>) -> Series {
    s.iter()
        .map(|v| swap.get(&v).unwrap().try_extract::<i32>().unwrap())
        .collect()
}

fn get_classes_swaps(classes: Vec<i32>) -> Vec<HashMap<i32, i32>> {
    classes
        .iter()
        .permutations(classes.len()) // permutations
        .collect::<Vec<_>>()
        .iter()
        .map(|permuted| {
            HashMap::<i32, i32>::from_iter(
                zip(&classes, permuted.clone()).map(|(a, b)| (a.clone(), b.clone())),
            )
        })
        .collect()
}

/// Classification results
/// 2 columns data frame.
///
/// | truth | prediction |
/// |-------|------------|
/// | ...   | ...        |
///
struct ClassificationResult(DataFrame);

impl ClassificationResult {
    fn classes(&self) -> Vec<AnyValue> {
        self.0
            .iter()
            .map(|s| s.iter().collect::<HashSet<_>>())
            .reduce(|acc, e| acc.union(&e).cloned().collect::<HashSet<_>>())
            .unwrap()
            .into_iter()
            .collect::<Vec<_>>()
    }

    fn tp(&self) -> usize {
        let truth = self.0.column("truth");
        let prediction = self.0.column("prediction");
        zip(truth, prediction)
            .filter(|(ref t, ref p)| t == p)
            .count()
    }

    /// Reassign classes to match truth
    ///
    /// clustering may assign classes that are not the original one.
    /// this function tries to match result classes to oringinal classes
    /// by swapping result classes
    fn reorder_classes(&self) -> Self {
        let classes = self.classes();
        let predicted = self.0.column("predicted");

        todo!()
    }

    /// Compute confusion matrix
    ///
    /// |                    |                        predicted                    |
    /// |                    | class A | class B | class C | ... | class N | total |
    /// |         | class A  |         |         |         | ... |         |       |
    /// |         | class B  |         |         |         | ... |         |       |
    /// | truth   | class C  |         |         |         | ... |         |       |
    /// |         | ...      | ...     | ...     | ...     | ... | ...     |       |
    /// |         | total    |         |         |         |     |         |       |
    fn confusion_matrix(&self) -> DataFrame {
        let classes = self.classes();
        let mut counts = HashMap::new(); // {(truth, predicted): count}
        for idx in 0..self.0.shape().0 {
            let row = self.0.get_row(idx).expect("idx is a valid index");
            *counts
                .entry((row.0[0].clone(), row.0[1].clone()))
                .or_insert(0) += 1;
        }
        let mut result = DataFrame::new(
            classes
                .clone()
                .into_iter()
                .map(|c| {
                    Series::new(
                        &format!("predicted\n{c}"),
                        classes
                            .clone()
                            .into_iter()
                            .map(|d| counts.get(&(d, c.clone())).unwrap_or(&0))
                            .copied()
                            .collect::<Vec<_>>(),
                    )
                })
                .collect(),
        )
        .unwrap();

        let mut idx_label = classes
            .iter()
            .map(|c| format!("truth\n{c}"))
            .collect::<Vec<_>>();
        let mut total =
            df!("total" => result.iter().map(|s| {s.iter().map(|c|c.try_extract::<i32>().unwrap()).sum::<i32>()})
                .collect::<Vec<_>>()
            )
            .unwrap()
            .transpose(None, None)
            .unwrap();
        dbg!(&result);
        dbg!(&total);
        let b = total.clone();
        let names =
            zip(b.iter().map(|c| c.name()), result.iter().map(|c| c.name())).collect::<Vec<_>>();
        for (old, new) in names {
            let total = total.rename(old, new).unwrap();
            dbg!(&total);
        }

        let mut result = result.vstack(&total).unwrap();
        result
            .insert_column(0, idx_label.into_iter().collect::<Series>())
            .unwrap();
        result
    }
}

/// kmeans
fn k_means(data: &DataFrame, n_cluster: usize) -> Result<KMeans<f64, L2Dist>> {
    let cols = data.get_column_names();
    let classes = data
        .get(cols.iter().position(|s| s == &"class").unwrap())
        .expect("classes should be provided");
    let data = data.drop("class")?;
    let data = DatasetBase::new(
        data.to_ndarray::<Float64Type>(IndexOrder::default())?,
        classes,
    );
    let model = KMeans::params(n_cluster).fit(&data).expect("data fitted");
    Ok(model)
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
    df.set_column_names(&columns)?;
    dbg!(&df);

    // rebuild description
    let descr = DataFramePreproc(df.clone()).describe()?;
    dbg!(descr);

    let kinds = [
        Normalization::Standard,
        Normalization::MinMax,
        Normalization::Quartiles,
        Normalization::Center,
    ];
    for kind in kinds {
        dbg!(kind);
        let normalized = DataFramePreproc(df.clone()).normalize(kind)?;
        dbg!(DataFramePreproc(normalized).describe()?);
    }

    let model = k_means(&df, 4)?;
    dbg!(&model);

    let pred = model.predict(DatasetBase::from(
        df.drop("class")?
            .to_ndarray::<Float64Type>(IndexOrder::default())?,
    ));
    dbg!(pred.targets());
    dbg!(pred.targets().iter().map(|&s| s as f64).collect::<Series>());
    dbg!(pred.targets().iter().collect::<Vec<_>>());
    dbg!(&df.column("class")?.clone().rename("truth"));
    let results = DataFrame::new(vec![
        df.column("class")?.clone().rename("truth").clone(),
        pred.targets()
            .iter()
            .map(|&s| s as i64)
            .collect::<Series>()
            .rename("prediction")
            .clone(),
    ])?;
    dbg!(&results);
    dbg!(ClassificationResult(results).confusion_matrix());

    Ok(())
}
