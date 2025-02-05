use crate::*;

#[derive(Debug)]
pub struct FlexBrain {
    speed_accel: f32,
    rotation_accel: f32,
    nv: Vec<nn::FlexNetwork>,//0 - сеть намерения (intention), 1 - двигательная (motive) сеть
}

impl FlexBrain {
    /// Выбранная по номеру нейронная сеть как хромосома
    pub(crate) fn as_chromosome(&self, nv_num: usize) -> ga::Chromosome {
        self.nv[nv_num].weights().collect()
    }
    /// Сеть - прямой проход
    pub(crate) fn propagate_0(&self,
                              vision_f: Vec<f32>,
                              vision_a: Vec<f32>) -> (f32, f32) {
        //соединяем 2 вектора
        let vision: Vec<f32> = vision_f.iter().cloned().chain(vision_a.iter().cloned()).collect();
        //расчет
        let response = self.nv[0].propagate(vision);

        let r0 = response[0].clamp(0.0, 1.0) - 0.5;
        let r1 = response[1].clamp(0.0, 1.0) - 0.5;
        let speed = (r0 + r1).clamp(-self.speed_accel, self.speed_accel);
        let rotation = (r0 - r1).clamp(-self.rotation_accel, self.rotation_accel);

        (speed, rotation)
    }
    // /// Сеть намерения (intention) - прямой проход
    // pub(crate) fn propagate_a(&self,
    //                           vision_f: Vec<f32>,
    //                           vision_a: Vec<f32>) -> (f32, f32, Vec<f32>) {
    //     //соединяем 2 вектора
    //     let vision: Vec<f32> = vision_f.iter().cloned().chain(vision_a.iter().cloned()).collect();
    //     //расчет
    //     let response = self.nv[0].propagate(vision);
    //     //получаем первые 2 в один срез, а в другой срез - остальные элементы
    //     let (resp1, resp2) = response.split_at(2);
    //
    //     let r0 = resp1[0].clamp(0.0, 1.0) - 0.5;
    //     let r1 = resp1[1].clamp(0.0, 1.0) - 0.5;
    //     let speed = (r0 + r1).clamp(-self.speed_accel, self.speed_accel);
    //     let rotation = (r0 - r1).clamp(-self.rotation_accel, self.rotation_accel);
    //
    //     (speed, rotation, resp2.to_vec())
    // }
    // /// Двигательная (motive) сеть - прямой проход
    // pub(crate) fn propagate_m(&self,
    //                           msgs: Vec<f32>,
    //                           speed: f32,
    //                           rotation: f32) -> (f32, f32) {
    //     let mut vision: Vec<f32> = Vec::from(msgs);
    //     vision.push(speed);
    //     vision.push(rotation);
    //     let response = self.nv[1].propagate(vision);
    //
    //     let r0 = response[0].clamp(0.0, 1.0) - 0.5;
    //     let r1 = response[1].clamp(0.0, 1.0) - 0.5;
    //     let speed = (r0 + r1).clamp(-self.speed_accel, self.speed_accel);
    //     let rotation = (r0 - r1).clamp(-self.rotation_accel, self.rotation_accel);
    //
    //     (speed, rotation)
    // }
}
impl FlexBrain {
    /// Конструктор
    fn new(config: &Config, nv: Vec<nn::FlexNetwork>) -> Self {
        Self {
            speed_accel: config.sim_speed_accel,
            rotation_accel: config.sim_rotation_accel,
            nv,//0 - сеть намерения (intention), 1 - двигательная (motive) сеть
        }
    }
    /// Brain из нейронных сетей с типовой топологией, но случайными весами
    pub(crate) fn random(config: &Config, rng: &mut dyn RngCore) -> Self {
        //сеть с типовой топологией, но случайными весами
        let mut nv: Vec<nn::FlexNetwork> = Vec::new();
        nv.push(nn::FlexNetwork::random(rng, &Self::topology_0(config)));
        // nv.push(nn::FlexNetwork::random(rng, &Self::topology_i(config)));
        // nv.push(nn::FlexNetwork::random(rng, &Self::topology_m(config)));
        Self::new(config, nv)
    }
    /// Brain из вектора хромосом каждой нейронной сети
    pub(crate) fn from_chromosome(config: &Config,
                                  chromosomes: Vec<ga::Chromosome>) -> Self {
        let mut nv: Vec<nn::FlexNetwork> = Vec::new();
        nv.push(nn::FlexNetwork::from_weights(chromosomes[0].iter()));
        // nv.push(nn::FlexNetwork::from_weights(chromosomes[1].iter()));
        Self::new(config, nv)
    }
    /// Типовая сеть
    fn topology_0(config: &Config) -> [nn::LayerTopologyFlex; 3] {
        //входной слой
        //(вес,слой,нейрон,вх.связь)
        let mut connections1: Vec<(f32, usize, usize, usize)> = Vec::new();
        for i in 1..=18 { // config.eye_cells = 9 * 2 вектора
            connections1.push((0.0,1,i,0));//bias
            connections1.push((1.0,1,i,i));//weights
        };
        //первый слой - 2 нейрона. 18 входов, 2 выхода
        //(вес,слой,нейрон,вх.связь) 2
        let mut connections2: Vec<(f32, usize, usize, usize)> = Vec::new();
        for i in 19..=20 { // 2
            connections2.push((0.0,2,i,0));//bias
            for j in 1..=18 {
                connections2.push((0.0,2,i,j));//weights
            }
        };
        //второй слой - 2 нейрона. 2 входа, 2 выхода
        //(вес,слой,нейрон,вх.связь) 2
        let mut connections3: Vec<(f32, usize, usize, usize)> = Vec::new();
        for i in 21..=22 { // 2
            connections3.push((0.0,3,i,0));//bias
            for j in 19..=20 {
                connections3.push((0.0,3,i,j));//weights
            }
        };

        [
            nn::LayerTopologyFlex {
                connections: connections1,
                activation: Activation::Relu,
            },
            nn::LayerTopologyFlex {
                connections: connections2,
                activation: Activation::Relu,
            },
            nn::LayerTopologyFlex {
                connections: connections3,
                activation: Activation::Relu,
            },
        ]
    }
    // /// Типовая сеть намерения (intention)
    // fn topology_i(config: &Config) -> [nn::FlexLayerTopology; 3] {
    //     //входной слой
    //     //(вес,слой,нейрон,вх.связь)
    //     let mut connections1: Vec<(f32, usize, usize, usize)> = Vec::new();
    //     for i in 1..=18 { // config.eye_cells = 9 * 2 вектора
    //         connections1.push((0.0,1,i,0));//bias
    //         connections1.push((1.0,1,i,i));//weights
    //     };
    //     //первый слой - 18 нейронов. 18 входов, 18 выходов
    //     //(вес,слой,нейрон,вх.связь) config.brain_neurons = 9 * 2 вектора
    //     let mut connections2: Vec<(f32, usize, usize, usize)> = Vec::new();
    //     for i in 19..=37 { // config.brain_neurons = 9 * 2 вектора
    //         connections2.push((0.0,2,i,0));//bias
    //         for j in 1..=18 {
    //             connections2.push((0.0,2,i,j));//weights
    //         }
    //     };
    //     //второй слой - 11 нейронов. 18 входов, 11 выходов
    //     //(вес,слой,нейрон,вх.связь) 2 + 9
    //     let mut connections3: Vec<(f32, usize, usize, usize)> = Vec::new();
    //     for i in 38..=48 { // 2 + 9
    //         connections3.push((0.0,3,i,0));//bias
    //         for j in 19..=37 {
    //             connections3.push((0.0,3,i,j));//weights
    //         }
    //     };
    //
    //     [
    //         nn::FlexLayerTopology {
    //             // neurons: config.eye_cells * 2,//=9 * 2 вектора
    //             connections: connections1,
    //             activation: Activation::Relu,
    //         },
    //         nn::FlexLayerTopology {
    //             // neurons: config.brain_neurons * 2,//=9 * 2 вектора
    //             connections: connections2,
    //             activation: Activation::Relu,
    //         },
    //         nn::FlexLayerTopology {
    //             // neurons: 2 + 9,
    //             connections: connections3,
    //             activation: Activation::Relu,
    //         },
    //     ]
    // }
    // /// Типовая двигательная (motive) сеть
    // fn topology_m(config: &Config) -> [nn::FlexLayerTopology; 3] {
    //     //входной слой
    //     //(вес,слой,нейрон,вх.связь)
    //     let mut connections1: Vec<(f32, usize, usize, usize)> = Vec::new();
    //     for i in 1..=11 { // config.eye_cells = 9 + 2 (speed, rotation)
    //         connections1.push((0.0,1,i,0));//bias
    //         connections1.push((1.0,1,i,i));//weights
    //     };
    //     //первый слой - 11 нейронов. 11 входов, 11 выходов
    //     //(вес,слой,нейрон,вх.связь) config.brain_neurons = 9 + 2 (speed, rotation)
    //     let mut connections2: Vec<(f32, usize, usize, usize)> = Vec::new();
    //     for i in 12..=23 { // config.brain_neurons = 9 + 2 (speed, rotation)
    //         connections2.push((0.0,2,i,0));//bias
    //         for j in 1..=11 {
    //             connections2.push((0.0,2,i,j));//weights
    //         }
    //     };
    //     //второй слой - 2 нейрона. 11 входов, 2 выхода
    //     //(вес,слой,нейрон,вх.связь) 2 + 9
    //     let mut connections3: Vec<(f32, usize, usize, usize)> = Vec::new();
    //     for i in 24..=25 { // 2
    //         connections3.push((0.0,3,i,0));//bias
    //         for j in 12..=23 {
    //             connections3.push((0.0,3,i,j));//weights
    //         }
    //     };
    //
    //     [
    //         nn::FlexLayerTopology {
    //             // neurons: config.eye_cells + 2, //=9 + 2
    //             connections: connections1,
    //             activation: Activation::Relu,
    //         },
    //         nn::FlexLayerTopology {
    //             // neurons: config.brain_neurons + 2, //=9 + 2
    //             connections: connections2,
    //             activation: Activation::Relu,
    //         },
    //         nn::FlexLayerTopology {
    //             // neurons: 2,
    //             connections: connections3,
    //             activation: Activation::Relu,
    //         },
    //     ]
    // }
}