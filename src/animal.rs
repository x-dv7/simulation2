use std::ops::Deref;
use crate::*;

#[derive(Debug)]
pub struct Animal {//птичка
    ///длительность жизни в эпохах (от 1 до 5 поколений, sim_generation_length/500)
    ///уменьшается с каждой эпохой
    pub(crate) generation_lifetime: usize,
    ///количество прожитых эпох
    pub(crate) generation_age: usize,
    ///позиция птички на карте
    pub(crate) position: na::Point2<f32>,
    ///поворот птички на карте
    pub(crate) rotation: na::Rotation2<f32>,
    ///обзор птички по еде (концентрация)
    pub(crate) vision_f: RefCell<Vec<f32>>,
    ///обзор птички по др. птичкам (номер наиболее концентрированной птички, концентрация)
    pub(crate) vision_a: RefCell<Vec<(usize, f32)>>,
    ///скорость птички
    pub(crate) speed: f32,
    ///Зрение
    pub(crate) eye: Eye,
    ///Мышление
    pub(crate) brain: FlexBrain,
    ///насыщенность за эпоху птички едой (кол. съеденного)
    pub(crate) satiation: usize,
    ///насыщенность, приведенная к animal.generation_age
    pub(crate) satiation_avg: f32,
}

impl Animal {
    pub fn position(&self) -> na::Point2<f32> {
        self.position
    }

    pub fn rotation(&self) -> na::Rotation2<f32> {
        self.rotation
    }

    pub fn vision(&self) -> Vec<f32> {//для Web-страницы, для отображение сектора обзора птичек &[f32]
        self.vision_f.borrow().clone()
    }
}

impl Animal {
    pub(crate) fn random(config: &Config, rng: &mut dyn RngCore) -> Self {
        let brain = FlexBrain::random(config, rng);

        Self::new(config, rng, brain)
    }

    pub(crate) fn from_chromosome(
        config: &Config,
        rng: &mut dyn RngCore,
        chromosomes: Vec<ga::Chromosome>,
    ) -> Self {
        let brain = FlexBrain::from_chromosome(config, chromosomes);

        Self::new(config, rng, brain)
    }
    //выбранная по номеру сеть как хромосома
    pub(crate) fn as_chromosome(&self, nv_num: usize) -> ga::Chromosome {
        self.brain.as_chromosome(nv_num)
    }
    //процесс видения еды и других птичек
    pub(crate) fn process_vision(&self,
                                 foods: &[Food],
                                 animals: &[Animal],
                                 cur: usize) -> (Vec<f32>, Vec<(usize, f32)>)
    {
        //смотрим на еду
        let vision_f =
            self.eye.process_vision_food(self.position, self.rotation, foods);
        //смотрим на других птичек
        let vision_a =
            self.eye.process_vision_animal(self.position, self.rotation, animals, cur);
        (vision_f, vision_a)
    }
    //обдумывание коммуникации и перемещения
    pub(crate) fn process_brain(&mut self, config: &Config) {
        let (vis_a_num, vis_a): (Vec<usize>, Vec<f32>) =
            self.vision_a.borrow().clone().into_iter().unzip();
        // //последовательная сеть размышлений и общения
        // //обдумывание положения птичек и еды -> сообщения другим птичкам и намерений по
        // //коррекции своего положения
        // let (speed, rotation, msgs) =
        //     self.brain.propagate_a(self.vision_f.borrow().clone(), vis_a);
        // // обдумывание сообщений птичек и намерений по своему положению -> коррекция положения
        // let (speed, rotation) = self.brain.propagate_m(msgs, speed, rotation);

        let (speed, rotation) =
            self.brain.propagate_0(self.vision_f.borrow().clone(), vis_a);

        //преобразование приращения положения в итоговое
        self.speed = (self.speed + speed).clamp(config.sim_speed_min, config.sim_speed_max);
        self.rotation = na::Rotation2::new(self.rotation.angle() + rotation);
    }
    //само перемещение
    pub(crate) fn process_movement(&mut self) {
        self.position += self.rotation * na::Vector2::new(0.0, self.speed);
        self.position.x = na::wrap(self.position.x, 0.0, 1.0);
        self.position.y = na::wrap(self.position.y, 0.0, 1.0);
    }
}

impl Animal {
    fn new(config: &Config, rng: &mut dyn RngCore, brain: FlexBrain) -> Self {
        Self {
            generation_lifetime: rng.gen_range(1..=config.sim_generation_length/500),//5
            generation_age: 0,
            position: rng.gen(),
            rotation: rng.gen(),
            vision_f: RefCell::new(vec![0.0; config.eye_cells]),
            vision_a: RefCell::new(vec![(0, 0.0); config.eye_cells]),
            speed: config.sim_speed_max,
            eye: Eye::new(config),
            brain,
            satiation: 0,
            satiation_avg: 0.0f32,
        }
    }
}
