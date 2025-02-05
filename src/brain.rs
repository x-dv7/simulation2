use crate::*;

#[derive(Debug)]
pub struct Brain {
    speed_accel: f32,
    rotation_accel: f32,
    nv: Vec<nn::SoftNetwork>,//0 - сеть намерения (intention), 1 - двигательная (motive) сеть
}

impl Brain {
    //сеть с типовой топологией, но случайными весами
    pub(crate) fn random(config: &Config, rng: &mut dyn RngCore) -> Self {
        //сеть с типовой топологией, но случайными весами
        let mut nv: Vec<nn::SoftNetwork> = Vec::new();
        nv.push(nn::SoftNetwork::random(rng, &Self::topology_i(config)));
        nv.push(nn::SoftNetwork::random(rng, &Self::topology_m(config)));
        Self::new(config, nv)
    }
    pub(crate) fn from_chromosome(config: &Config,
                                  chromosomes: Vec<ga::Chromosome>) -> Self {
        let mut nv: Vec<nn::SoftNetwork> = Vec::new();
        nv.push(nn::SoftNetwork::from_weights(chromosomes[0].iter()));
        nv.push(nn::SoftNetwork::from_weights(chromosomes[1].iter()));
        Self::new(config, nv)
    }
    //выбранная по номеру сеть как хромосома
    pub(crate) fn as_chromosome(&self, nv_num: usize) -> ga::Chromosome {
        self.nv[nv_num].weights().collect()
    }
    //сеть намерения (intention) - прямой проход
    pub(crate) fn propagate_a(&self,
                              vision_f: Vec<f32>,
                              vision_a: Vec<f32>) -> (f32, f32, Vec<f32>) {
        //соединяем 2 вектора
        let vision: Vec<f32> = vision_f.iter().cloned().chain(vision_a.iter().cloned()).collect();
        //расчет
        let response = self.nv[0].propagate(vision);
        //получаем первые 2 в один срез, а в другой срез - остальные элементы
        let (resp1, resp2) = response.split_at(2);

        let r0 = resp1[0].clamp(0.0, 1.0) - 0.5;
        let r1 = resp1[1].clamp(0.0, 1.0) - 0.5;
        let speed = (r0 + r1).clamp(-self.speed_accel, self.speed_accel);
        let rotation = (r0 - r1).clamp(-self.rotation_accel, self.rotation_accel);

        (speed, rotation, resp2.to_vec())
    }
    //двигательная (motive) сеть - прямой проход
    pub(crate) fn propagate_m(&self,
                              msgs: Vec<f32>,
                              speed: f32,
                              rotation: f32) -> (f32, f32) {
        let mut vision: Vec<f32> = Vec::from(msgs);
        vision.push(speed);
        vision.push(rotation);
        let response = self.nv[1].propagate(vision);

        let r0 = response[0].clamp(0.0, 1.0) - 0.5;
        let r1 = response[1].clamp(0.0, 1.0) - 0.5;
        let speed = (r0 + r1).clamp(-self.speed_accel, self.speed_accel);
        let rotation = (r0 - r1).clamp(-self.rotation_accel, self.rotation_accel);

        (speed, rotation)
    }

}

impl Brain {
    fn new(config: &Config, nv: Vec<nn::SoftNetwork>) -> Self {
        Self {
            speed_accel: config.sim_speed_accel,
            rotation_accel: config.sim_rotation_accel,
            nv,//0 - сеть намерения (intention), 1 - двигательная (motive) сеть
        }
    }
    //типовая сеть намерения (intention)
    fn topology_i(config: &Config) -> [nn::SoftLayerTopology; 3] {
        let mut connections1: Vec<(usize, usize)> = Vec::new();
        // for i in 1..=9 { // config.eye_cells = 9
        for i in 1..=18 { // config.eye_cells = 9 * 2 вектора
            connections1.push((i, i));
        };
        let mut connections2: Vec<(usize, usize)> = Vec::new();
        // for i in 10..=19 { // config.brain_neurons = 9
        //      for j in 1..=9 {
        for i in 19..=37 { // config.brain_neurons = 9 * 2 вектора
            for j in 1..=18 {
                connections2.push((i, j));
            }
        };
        let mut connections3: Vec<(usize, usize)> = Vec::new();
        // for i in 20..=21 { // 2
        //     for j in 10..=19 {
        for i in 38..=48 { // 2 + 9
            for j in 19..=37 {
                connections3.push((i, j));
            }
        };

        [
            nn::SoftLayerTopology {
                neurons: config.eye_cells * 2,//=9 * 2 вектора
                connections: connections1,
            },
            nn::SoftLayerTopology {
                neurons: config.brain_neurons * 2,//=9 * 2 вектора
                connections: connections2,
            },
            nn::SoftLayerTopology {
                neurons: 2 + 9,
                connections: connections3,
            },
        ]
    }
    //типовая двигательная (motive) сеть
    fn topology_m(config: &Config) -> [nn::SoftLayerTopology; 3] {
        let mut connections1: Vec<(usize, usize)> = Vec::new();
        for i in 1..=11 { // config.eye_cells = 9 + 2 (speed, rotation)
            connections1.push((i, i));
        };
        let mut connections2: Vec<(usize, usize)> = Vec::new();
        for i in 12..=23 { // config.brain_neurons = 9 + 2 (speed, rotation)
            for j in 1..=11 {
                connections2.push((i, j));
            }
        };
        let mut connections3: Vec<(usize, usize)> = Vec::new();
        for i in 24..=25 { // 2
            for j in 12..=23 {
                connections3.push((i, j));
            }
        };

        [
            nn::SoftLayerTopology {
                neurons: config.eye_cells + 2, //=9
                connections: connections1,
            },
            nn::SoftLayerTopology {
                neurons: config.brain_neurons + 2, //=9
                connections: connections2,
            },
            nn::SoftLayerTopology {
                neurons: 2,
                connections: connections3,
            },
        ]
    }
}