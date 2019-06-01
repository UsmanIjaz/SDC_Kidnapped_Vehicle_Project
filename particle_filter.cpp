/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#include "helper_functions.h"


using namespace std;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 50;
	
	default_random_engine gen;
	
	normal_distribution<double> N_x(x,std[0]);
	normal_distribution<double> N_y(y,std[1]);
	normal_distribution<double> N_theta(theta,std[2]);
	
	for(int i=0; i<num_particles;i++)
	{
		Particle particle;
		particle.id = i;
		particle.x = N_x(gen);
		particle.y = N_y(gen);
		particle.theta = N_theta(gen);
		particle.weight = 1.0;
		
		particles.push_back(particle);
		weights.push_back(1.0);
	}
	
	is_initialized = true;
	
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	default_random_engine gen;
	
	for(int i = 0;i < num_particles ; i++){
		
		double new_x ;
		double new_y ;
		double new_theta ;
		
		if(yaw_rate==0){
			new_x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
			new_y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
			new_theta = particles[i].theta;
		}
		else{
			new_x = particles[i].x + velocity/yaw_rate * (sin(particles[i].theta+yaw_rate*delta_t) - sin(particles[i].theta));
			new_y = particles[i].y + velocity/yaw_rate * (cos(particles[i].theta)-cos(particles[i].theta+yaw_rate*delta_t));
			new_theta = particles[i].theta+yaw_rate*delta_t;
		}
		
		normal_distribution<double> N_x(new_x,std_pos[0]);
		normal_distribution<double> N_y(new_y,std_pos[1]);
		normal_distribution<double> N_theta(new_theta,std_pos[2]);
		
		
		particles[i].x = N_x(gen);
		particles[i].y = N_y(gen);
		particles[i].theta = N_theta(gen);
	
	
	}

}


LandmarkObs transformation(Particle particle, LandmarkObs observation) {
    LandmarkObs transformed_observation;

    transformed_observation.id = observation.id;
    transformed_observation.x = particle.x + (observation.x * cos(particle.theta)) - (observation.y * sin(particle.theta));
    transformed_observation.y = particle.y + (observation.x * sin(particle.theta)) + (observation.y * cos(particle.theta));

    return transformed_observation;
}

int ParticleFilter::dataAssociation(LandmarkObs obs, Map map_landmarks) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
	double nearest_distance = 1000000.0;
	int nearest_id = -1;
	
	for(int i =0; i<map_landmarks.landmark_list.size(); i++)
	{
	
		Map:: single_landmark_s temp = map_landmarks.landmark_list[i];
		double distance = dist(obs.x,obs.y,temp.x_f,temp.y_f);
		if(distance < nearest_distance)	
		{
			nearest_distance = distance;
			nearest_id = i;
		}
	}
	return nearest_id;

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	
	// gaussian distribution parameters
	
	for(int i = 0; i <num_particles; i++ )
	{
	
		double weight = 1.0;
		vector<double> s_x;
		vector<double> s_y;
		vector<int> associations;
		
		Particle particle = particles[i];
		
		for(int j = 0; j< observations.size(); j++)
		{		
			LandmarkObs lm_obs = observations[j];
			LandmarkObs tf_obs = transformation(particle,lm_obs);
			
			tf_obs.id = dataAssociation(tf_obs,map_landmarks);
			Map:: single_landmark_s predicted = map_landmarks.landmark_list[tf_obs.id];

			double exp_arg = -0.5 * ( pow((tf_obs.x-predicted.x_f),2)/pow(std_landmark[0],2)+ pow((tf_obs.y-predicted.y_f),2)/pow(std_landmark[1],2) );
			weight = weight * exp(exp_arg)/(2.0*M_PI*std_landmark[0]*std_landmark[1]);	
		}
		
		particle = SetAssociations(particle,associations,s_x,s_y);
		particle.weight = weight;
		
		particles[i] = particle;
		weights[i] = weight;
	}
	


}



void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	default_random_engine gen;
	discrete_distribution<int> dist(weights.begin(),weights.end());
	
	vector<Particle> resampled_particles;
	
	for(int i = 0 ;i < num_particles; i++)
		resampled_particles.push_back(particles[dist(gen)]);
	
	particles =  resampled_particles;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
