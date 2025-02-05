//lifelong learning

mod food;
mod animal;
mod animal_individual;
mod config;
mod world;
mod eye;
// mod brain;
mod flex_brain;
mod statistics;


pub use self::food::*;
pub use self::animal::*;
pub use self::animal_individual::*;
pub use self::config::*;
pub use self::world::*;
pub use self::eye::*;
// pub use self::brain::*;
pub use self::flex_brain::*;
pub use self::statistics::*;


use rand::{Rng, RngCore};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::f32::consts::*;
use std::fmt;
use std::cell::RefCell;
use lib_genetic_algorithm::{Individual, IndividualFlex};
use rand::rngs::OsRng;
use candle_nn::{Activation};
use {lib_genetic_algorithm as ga, lib_neural_network as nn, nalgebra as na};

pub struct Simulation {
    /// генератор случайных значений
    rng: OsRng,
    /// текущая конфигурация
    config: Config,
    /// Мир симуляции, состоит из птичек и еды
    world: World,
    /// номер эпохи обучения, растет неограниченно
    generation: usize,
    /// номер шага симуляции, растет до config.sim_generation_length
    age: usize,
}

impl Simulation {
    pub fn random(config: Config) -> Self {
        // Создаем новый генератор
        let mut rng = OsRng::default();
        let world = World::random(&config, &mut rng);

        Self {
            rng,
            config,
            world,
            generation: 0,
            age: 0,
        }
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn set_config(&mut self, conf: Config) {
        self.config = conf;//заменим конфигурацию на новую
        //заменим птиц и еду на новую
        self.world = World::random(&self.config, &mut self.rng);
    }

    pub fn world(&self) -> &World {
        &self.world
    }

    pub fn step(&mut self) -> Option<Statistics> {
        self.process_collisions();//столкновения с едой
        self.process_brains();//общение между птичками и обдумывание перемещения
        self.process_movements();//само перемещение
        self.try_evolving()//обучение новой популяции
    }
}

impl Simulation {
    //обработка столкновения с едой
    fn process_collisions(&mut self) {
        for animal in &mut self.world.animals {
            for food in &mut self.world.foods {
                let distance = na::distance(&animal.position, &food.position);

                if distance <= self.config.food_size {
                    animal.satiation += 1;//насыщенность за эпоху
                    food.position = self.rng.gen();//новая еда
                }
            }
        }
    }
    //обдумывание перемещения
    fn process_brains(&mut self) {
        //птички сканируют пространство
        for (j, animal) in self.world.animals.iter().enumerate() {
            let (vf, va) =
                animal.process_vision(&self.world.foods, &self.world.animals, j);
            *(animal.vision_f.borrow_mut()) = vf;
            *(animal.vision_a.borrow_mut()) = va;
        }
        //птички обдумывают
        for animal in &mut self.world.animals {
            animal.process_brain(&self.config);
        }
    }
    //само перемещение
    fn process_movements(&mut self) {
        for animal in &mut self.world.animals {
            animal.process_movement();
        }
    }
    //обучение новых птичек при накоплении достаточного возраста
    fn try_evolving(&mut self) -> Option<Statistics> {
        self.age += 1;
        if self.age > self.config.sim_generation_length / 4 {
            //каждый 625 шаг меняется поколение и те птички, у которых lifetime уменьшился
            //до 0 заменяются
            Some(self.evolve())
        }
        else {
            None
        }
    }
    //само обучение одной эпохи эволюции/замена некоторых птичек на новых
    fn evolve(&mut self) -> Statistics {
        self.age = 0;
        self.generation += 1; //увеличивает номер поколения
        //средняя насыщенность каждой птички за весь её возраст (используется при обучении в ГА)
        for animal in &mut self.world.animals {
            animal.generation_age += 1;//возраст птички в эпохах
            //средняя насыщенность птички за весь её возраст
            animal.satiation_avg =
                (animal.satiation_avg * (animal.generation_age -1) as f32
                + animal.satiation as f32)
                / animal.generation_age as f32;
        }
        //Создаем генетический алгоритм
        let ga = ga::GeneticFlexAlgorithm::new(
            self.config.sim_generation_length,//для генерации времени жизни
            ga::RouletteWheelSelection,
            ga::UniformCrossover,
            ga::Flex1Mutation::new(self.config.ga_mut_chance,
                                   self.config.ga_mut_coeff,
                                   self.config.eye_cells),//для мутации кол. входов
        );
        // // сортируем птичек с минимальным насыщением (в самом верху)
        // self.world.animals.sort_by(|a, b| a.satiation.cmp(&b.satiation));
        //Получаем (0 - сеть намерения (intention)):
        // - все птички в виде AnimalIndividual
        let all_i: Vec<AnimalIndividual> = self
            .world
            .animals
            .iter()
            .map(|a| AnimalIndividual::from_animal(a,0))
            .collect();
        // //Получаем (1 - двигательная (motive) сеть):
        // // - все птички в виде AnimalIndividual
        // let all_m: Vec<AnimalIndividual> = self
        //     .world
        //     .animals
        //     .iter()
        //     .map(|a| AnimalIndividual::from_animal(a,1))
        //     .collect();
        //Шаг эволюции
        //На входе: генератор случайных значений и список всех птичек в виде AnimalIndividual
        //На выходе: новая популяция птичек (AnimalIndividual)
        //и статистика по прошлой популяции
        let (individuals_i, stats_i) =
            ga.evolve(&mut self.rng, &all_i);
        // let (individuals_m, stats_m) =
        //     ga.evolve_1(&mut self.rng, &all_m);
        //Замена птичек в популяции
        for (j, animal) in self.world.animals.iter_mut().enumerate() {
            //время жизни в поколениях уменьшаем для "плохих" птичек
            //при этом "хорошие" птички сохраняют свою жизнь дольше
            if individuals_i[j].changed() == true {//под замену
                let chrs: Vec<ga::Chromosome> = vec![individuals_i[j].chromosome().clone()];//,
                                                     // individuals_m[j].chromosome().clone()];
                *animal = Animal::from_chromosome(&self.config, &mut self.rng, chrs);
            } else {//замены нет
                animal.satiation = 0;
            };
            animal.generation_lifetime = individuals_i[j].life_time();
        }
        //Статистика по прошлой популяции
        Statistics {
            generation: self.generation - 1,
            ga: vec![stats_i], //stats_m],
        }
    }
}

impl Simulation {
    pub fn train(&mut self) -> Statistics { //, rng: &mut dyn RngCore
        loop {
            if let Some(statistics) = self.step() { //rng
                return statistics;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // use rand::SeedableRng;
    // use rand_chacha::ChaCha8Rng;

    #[test]
    #[ignore]
    fn test() {
        // let mut rng = ChaCha8Rng::from_seed(Default::default());
        let mut sim = Simulation::random(Default::default());//, &mut rng

        let avg_fitness = (0..10)
            .map(|_| sim.train().ga[0].avg_fitness())//&mut rng
            .sum::<f32>()
            / 10.0;

        approx::assert_relative_eq!(31.944998, avg_fitness);
    }
}