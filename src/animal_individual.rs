use lib_genetic_algorithm::Chromosome;
use crate::*;
#[derive(Clone, Debug)]
pub struct AnimalIndividual {
    fitness: f32,//насыщенность птички едой
    chromosome: ga::Chromosome,
    life_time: usize,//life_time сколько осталось жить птичке
    changed: bool,//замененная птичка
    mut_force: usize,//сила мутации
}

impl AnimalIndividual {
    //выбранная по номеру сеть как хромосома
    pub fn from_animal(animal: &Animal, nv_num: usize) -> Self {//из птички в хромосому
        Self {
            fitness: animal.satiation_avg,
            chromosome: animal.as_chromosome(nv_num),
            life_time: animal.generation_lifetime,
            changed: false,
            mut_force: 1,
        }
    }
}

impl ga::Individual for AnimalIndividual {
    fn create(chromosome: ga::Chromosome) -> Self {
        Self {
            fitness: 0.0,
            chromosome,
            life_time: 0,
            changed: false,
            mut_force: 1,
        }
    }

    fn chromosome(&self) -> &ga::Chromosome {
        &self.chromosome
    }
    fn chromosome_mut(&mut self) -> &mut ga::Chromosome {
        &mut self.chromosome
    }
    fn fitness(&self) -> f32 {
        self.fitness
    }
}

impl ga::IndividualFlex for AnimalIndividual {
    fn create(chromosome: Chromosome, life_time: usize, changed: bool, mut_force: usize) -> Self  {
        Self {
            fitness: 0.0,
            chromosome,
            life_time,
            changed,
            mut_force,
        }
    }
    fn life_time(&self) ->  usize {//life_time сколько осталось жить птичке
        self.life_time
    }
    fn changed(&self) -> bool {//замененная птичка
        self.changed
    }
    //сила мутации
    fn mut_force(&self) -> usize {
        self.mut_force
    }
}