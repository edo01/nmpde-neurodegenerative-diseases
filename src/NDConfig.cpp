#include "NDConfig.hpp"

void NDConfig::parse(int argc, char *argv[])
{
  int opt;
    while((opt = getopt(argc, argv, "d:T:a:t:e:x:m:g:h:D:o:s:c:f:")) != -1)
    {
      switch(opt) 
      {
        case 'D': this->dim = std::atoi(optarg); break;
        case 'T': this->T = std::atof(optarg); break;
        case 'a': this->alpha = std::atof(optarg); break;
        case 't': this->deltat = std::atof(optarg); break;
        case 'g': this->degree = std::atoi(optarg); break;
        case 'e': this->d_ext = std::atof(optarg); break;
        case 'x': this->d_axn = std::atof(optarg); break;
        case 'c': this->C_0 = std::atof(optarg); break;
        case 'm': this->mesh = optarg; break;
        case 'o': this->output_filename = optarg; break;
        case 'd': this->output_dir = optarg; break;
        case 's': this->seeding_region_type = SeedingRegionType(std::atoi(optarg)); break;
        case 'f': this->fiber_field_type = FiberFieldType(std::atoi(optarg)); break;
        case 'h':
            std::cerr << "Usage: " << argv[0] << " [options]\n";
            exit(EXIT_SUCCESS);
        default:
            std::cerr << "Invalid option\n";
            exit(EXIT_FAILURE);
      }
    }
}