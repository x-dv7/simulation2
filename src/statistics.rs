use crate::*;

#[derive(Clone, Debug)]
pub struct Statistics {
    pub generation: usize,
    pub ga: Vec<ga::Statistics>,
}

impl fmt::Display for Statistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Поколение {}:", self.generation)?;
        writeln!(f, "Макс.Нейрон: {}", self.ga[0].max_neuron_num())?;
        writeln!(f, "Сети: {}", json!(&self.ga[0].neurons_by_layer()))?;
        write!(
            f,
            "min[{:.2}] max[{:.2}] avg[{:.2}] median[{:.2}] изм[{:.2}];",
            &self.ga[0].min_fitness(),
            &self.ga[0].max_fitness(),
            &self.ga[0].avg_fitness(),
            &self.ga[0].median_fitness(),
            &self.ga[0].changed_count()
        )?;

        // for ga1 in &self.ga {
        //     write!(
        //         f,
        //         "min[{:.2}] max[{:.2}] avg[{:.2}] median[{:.2}]; ",
        //         ga1.min_fitness(),
        //         ga1.max_fitness(),
        //         ga1.avg_fitness(),
        //         ga1.median_fitness())?
        // }
        Ok(())
    }
}