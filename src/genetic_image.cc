#include <iostream>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <jpeglib.h>
#include <stdlib.h>
#include <random>
#include <time.h>
#include <ctime>
#include <thread>
#include <omp.h>

#define thread_local __thread
#define POPULATION_SIZE 10000
#define MAX_GENERATIONS 1000

unsigned char minimum = 0, maximum = 255;
int compressionRate = 10;

using namespace std; 

typedef struct Chromosome
{
    unsigned char* data;
    int size;
} Chromosome;

typedef struct TImage
{
  int width;
  int height;
  int bytes_per_pixel;
  J_COLOR_SPACE color_space;
  int size;
  unsigned char *data;
} TImage;

int read_jpeg_file( char *filename, TImage *img )
{
    /* these are standard libjpeg structures for reading(decompression) */
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    /* libjpeg data structure for storing one row, that is, scanline of an image */
    JSAMPROW row_pointer[1];
    FILE *infile = fopen( filename, "rb" );
    unsigned long location = 0;
    int i = 0;
   
    if ( !infile )
    {
      printf("Error opening jpeg file %s\n!", filename );
      return -1;
    }
    /* here we set up the standard libjpeg error handler */
    cinfo.err = jpeg_std_error( &jerr );
    /* setup decompression process and source, then read JPEG header */
    jpeg_create_decompress( &cinfo );
    /* this makes the library read from infile */
    jpeg_stdio_src( &cinfo, infile );
    /* reading the image header which contains image information */
    jpeg_read_header( &cinfo, TRUE );
    /* Uncomment the following to output image information, if needed. */
    img->width=cinfo.image_width;
    img->height=cinfo.image_height;
    img->bytes_per_pixel=cinfo.num_components;
    img->color_space=cinfo.jpeg_color_space;

    /* Start decompression jpeg here */
    jpeg_start_decompress( &cinfo );

    /* allocate memory to hold the uncompressed image */
    img->size=cinfo.output_width*cinfo.output_height*cinfo.num_components;
    img->data = (unsigned char*)malloc( img->size );
    /* now actually read the jpeg into the raw buffer */
    row_pointer[0] = (unsigned char *)malloc( cinfo.output_width*cinfo.num_components );
    /* read one scan line at a time */
    while( cinfo.output_scanline < cinfo.image_height )
    {
      jpeg_read_scanlines( &cinfo, row_pointer, 1 );
      for( i=0; i<cinfo.image_width*cinfo.num_components;i++)
        {
          img->data[location++] = row_pointer[0][i];
        }
    }
    /* wrap up decompression, destroy objects, free pointers and close open files */
    jpeg_finish_decompress( &cinfo );
    jpeg_destroy_decompress( &cinfo );
    free( row_pointer[0] );
    fclose( infile );
    /* yup, we succeeded! */
    return 1;
}

/**
 * write_jpeg_file Writes the raw image data stored in the raw_image buffer
 * to a jpeg image with default compression and smoothing options in the file
 * specified by *filename.
 *
 * \returns positive integer if successful, -1 otherwise
 * \param *filename char string specifying the file name to save to
 *
 */
int write_jpeg_file( char *filename, TImage *img )
{
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
   
    /* this is a pointer to one row of image data */
    JSAMPROW row_pointer[1];
    FILE *outfile = fopen( filename, "wb" );
   
    if ( !outfile )
    {
        printf("Error opening output jpeg file %s\n!", filename );
        return -1;
    }
    cinfo.err = jpeg_std_error( &jerr );
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);

    /* Setting the parameters of the output file here */
    cinfo.image_width = img->width;
    cinfo.image_height = img->height;
    cinfo.input_components = img->bytes_per_pixel;
    cinfo.in_color_space = img->color_space;
    /* default compression parameters, we shouldn't be worried about these */
    jpeg_set_defaults( &cinfo );
    /* Now do the compression .. */
    jpeg_start_compress( &cinfo, TRUE );
    /* like reading a file, this time write one row at a time */
    while( cinfo.next_scanline < cinfo.image_height )
    {
        row_pointer[0] = &img->data[ cinfo.next_scanline * cinfo.image_width *  cinfo.input_components];
        jpeg_write_scanlines( &cinfo, row_pointer, 1 );
    }
    /* similar to read file, clean up after we're done compressing */
    jpeg_finish_compress( &cinfo );
    jpeg_destroy_compress( &cinfo );
    fclose( outfile );
    /* success code is 1! */
    return 1;
}

/// <summary>
/// Gets the minimum and maximum value of an image data.
/// </summary>
/// <param name="image">The image.</param>
/// <param name="min">A reference to an unsigned char variable to store the minimum value.</param>
/// <param name="max">A reference to an unsigned char variable to store the maximum value.</param>
void getMinMax(TImage* image, unsigned char &min, unsigned char &max)
{
    min = 255;
    max = 0;
    for (int i=0; i<image->size; i++)
    {
        if(image->data[i] > max)
        {
            max = image->data[i];
        }
        if(image->data[i] < min)
        {
            min = image->data[i];
        }
    }
}

/// <summary>
/// Compresses the data of an image with the specified compression rate.
/// </summary>
/// <param name="image">The image.</param>
/// <param name="compressionRate">A value indicating the compression rate.</param>
void compressImage(TImage* image, double compressionRate)
{
    for (int i=0; i<image->size; i++)
    {
        image->data[i] = image->data[i] * compressionRate;
    }
}

/// <summary>
/// Gets a random number between specified minimum and maximum values.
/// </summary>
/// <param name="min">The minimum value.</param>
/// <param name="max">The maximum value.</param>
/// <returns>
/// A random integer between min and max values.
/// </returns> 
int randomNumber(const int& min, const int& max) 
{
    static thread_local mt19937* generator = nullptr;

    std::hash<std::thread::id> hasher;
    if (!generator) generator = new mt19937(clock() + hasher(this_thread::get_id()));
    uniform_int_distribution<int> distribution(min, max);
    
    return distribution(*generator);
}

/// <summary>
/// Gets a random mutated gene.
/// </summary>
/// <returns>
/// A random mutated gene.
/// </returns> 
unsigned char mutateGene() 
{ 
	int r = randomNumber(minimum, maximum); 

	return (unsigned char)r; 
} 

/// <summary>
/// Creates a random chromosome with the specified size.
/// </summary>
/// <param name="size">The size of the chromosome.</param>
/// <returns>
/// A random chromosome.
/// </returns> 
Chromosome createChromosome(int size) 
{ 
    Chromosome chromosome;
    chromosome.data = new unsigned char[size];
    chromosome.size = size;

    for(int i = 0; i < size; i++)
    {
        chromosome.data[i] = mutateGene();
    }

	return chromosome; 
} 

/// <summary>
/// A class that represents an individual member of the population.
/// </summary>
class Individual 
{ 
public: 
    /// <summary>
    /// The chromosome of this individual.
    /// </summary>
	Chromosome chromosome;
 
    /// <summary>
    /// The fitness of this individual.
    /// </summary>
	int fitness; 
    
    /// <summary>
    /// Initializes a new instance of Individual.
    /// </summary>
    Individual();
    
    /// <summary>
    /// Initializes a new instance of Individual.
    /// </summary>
    /// <param name="chromosome">The chromosome for the individual.</param>
	Individual(Chromosome &chromosome); 

    /// <summary>
    /// Destroys this instance of Individual.
    /// </summary>
    ~Individual();

    /// <summary>
    /// Creates a new Individual resulting of the combination of this instance and another Individual.
    /// </summary>
    /// <param name="parent2">The other Individual to create the combination.</param>
	Individual mate(const Individual &parent2) const; 

    /// <summary>
    /// Calculates the fitness value of this Individual's chromosome.
    /// </summary>
    /// <param name="original">The original image to calculate the fitness value.</param>
	int calculateFitness(TImage* original); 

    /// <summary>
    /// Frees the memory of this instances's chromosome.
    /// </summary>
    void freeChromosome();
};

/// <summary>
/// Frees the memory of this instances's chromosome.
/// </summary>
void Individual::freeChromosome()
{
    if(this->chromosome.size > 0)
    {
        delete [] this->chromosome.data;
        this->chromosome.size = 0;
    }
}

/// <summary>
/// Initializes a new instance of Individual.
/// </summary>
Individual::Individual()
{
};

/// <summary>
/// Initializes a new instance of Individual.
/// </summary>
/// <param name="chromosome">The chromosome for the individual.</param>
Individual::Individual(Chromosome &chromosome) 
{ 
	this->chromosome = chromosome; 
	fitness = 0;
};

/// <summary>
/// Destroys this instance of Individual.
/// </summary>
Individual::~Individual()
{
}

/// <summary>
/// Creates a new Individual resulting of the combination of this instance and another Individual.
/// </summary>
/// <param name="parent2">The other Individual to create the combination.</param>
/// <return>A new individual resulting of the combination of two individuals.</return>
Individual Individual::mate(const Individual &parent2) const
{ 
    const double p1Chance = 0.45;
    const double p2Chance = 0.9;
	// chromosome for offspring 
	Chromosome child_chromosome;
    child_chromosome.data = new unsigned char[this->chromosome.size];
    child_chromosome.size = this->chromosome.size; 

    for(int i=0;i<this->chromosome.size;i++)
    {
	    // random probability 
	    float p = randomNumber(0, 100)/100.0; 

	    // if prob is less than 0.45, insert gene 
	    // from parent 1 
	    if(p < p1Chance) 
		    child_chromosome.data[i] = this->chromosome.data[i]; 

	    // if prob is between 0.45 and 0.90, insert 
	    // gene from parent 2 
	    else if(p < p2Chance) 
		    child_chromosome.data[i] = parent2.chromosome.data[i]; 

	    // otherwise insert random gene(mutate), 
	    // for maintaining diversity 
	    else
		    child_chromosome.data[i] = mutateGene(); 
    }

	// create new Individual(offspring) using 
	// generated chromosome for offspring 
	return Individual(child_chromosome); 
}; 

/// <summary>
/// Creates a new image from the data of the specified chromosome.
/// </summary>
/// <param name="original">The original image to obtain the parameters.</param>
/// <param name="chromosome">The chromosome with the image data.</param>
/// <return>An image struct with the chromosome as data.</return>
TImage getImageFromChromosome(TImage* original, Chromosome chromosome)
{
    TImage image;
    image.width = original->width;
    image.height = original->height;
    image.bytes_per_pixel = original->bytes_per_pixel;
    image.color_space = original->color_space;
    image.size = original->size;
    image.data = new unsigned char[original->size];

    for (int i=0; i<original->size; i++)
    {
        image.data[i] = chromosome.data[i];
    }

    return image;
}

/// <summary>
/// Calculates the fitness value of this Individual's chromosome.
/// </summary>
/// <param name="original">The original image to calculate the fitness value.</param>
/// <return>The fitness of this individual.</return>
int Individual::calculateFitness(TImage* original)
{
    int diff = 0;
    for(int i = 0; i < original->size; i++)
    {
        int diff1 = original->data[i] - this->chromosome.data[i];

        if(diff1 != 0)
            diff++;
    }

    this->fitness = diff;

	return diff;	 
}; 

/// <summary>
/// Determines if the fitness of an instance of Individual is less than another.
/// </summary>
/// <param name="ind1">The first Individual.</param>
/// <param name="ind2">The second Individual.</param>
/// <return>
/// True if first individual fitness is less than second individual fitness, false otherwise.
/// </return>
bool operator<(const Individual &ind1, const Individual &ind2) 
{ 
	return ind1.fitness < ind2.fitness; 
} 

/// <summary>
/// Performs the genetic algorithm.
/// </summary>
int main() 
{ 
    char *infilename = "image.jpg";
    TImage original;
    if(read_jpeg_file(infilename, &original) <= 0)
    {
        cout<<"Image could not be read"<<endl;
    }
    compressImage(&original, 1/(double)compressionRate);
    getMinMax(&original, minimum, maximum);
    cout<<"Min: "<<(int)minimum<<" Max: "<<(int)maximum<<endl;

	// Current generation 
	int generation = 0; 
    int elite = (10 * POPULATION_SIZE) / 100; 

	vector<Individual> population; 
	bool found = false; 

    cout<<"Creating initial population"<<endl;
	// Create initial population 
	for(int i = 0; i < POPULATION_SIZE; i++) 
	{ 
		Chromosome gnome = createChromosome(original.size);
        Individual individual(gnome);
        individual.calculateFitness(&original); 
		population.push_back(individual); 
	} 

	while(!found && generation < MAX_GENERATIONS)
	{ 
        #if defined(_OPENMP)
        double start = omp_get_wtime(); 
        #else
        clock_t begin = clock();
        #endif

		// Sort the population in increasing order of fitness score.
		sort(population.begin(), population.end()); 

		// If best Individual's fitness is 0 then we have reach our target.
		if(population[0].fitness <= 0) 
		{ 
			found = true; 

			break; 
		} 

        cout<<"Creating new generation"<<endl;
		vector<Individual> new_generation; 

		// Perform elitism, that means 10% of fittest population goes directly to next generation.
		for(int i = 0; i < elite; i++)
        { 
			new_generation.push_back(population[i]); 
        }

		// From 50% of fittest population, Individuals will mate to produce offspring.
	    for(int i = elite; i < population.size(); i++) 
	    { 
		    int len = population.size(); 
            Individual parent1, parent2;
	        int r = randomNumber(0, len / 2); 
	        parent1 = population[r]; 
	        r = randomNumber(0, len / 2); 
	        parent2 = population[r]; 
		    Individual offspring = parent1.mate(parent2); 
            offspring.calculateFitness(&original);

		    new_generation.push_back(offspring);
	    }

        // Free memory of parents.
        for(int i = elite; i < population.size(); i++)
        {
            population[i].freeChromosome();
        }
		population = new_generation; 

        // Calculate average fitness of new generation.
        long avgFitness = 0;
        for(int k = 0; k < population.size(); k++)
        {
            avgFitness += population[k].fitness;
        }
        avgFitness /= population.size();
        
        double elapsedSeconds = 0.0;
        #if defined(_OPENMP)
        elapsedSeconds = omp_get_wtime() - start;
        #else
        elapsedSeconds = double(clock() - begin) / CLOCKS_PER_SEC;
        #endif

		cout<<"Generation: "<<generation<<endl; 
		cout<<"Fitness: "<<population[0].fitness<<endl;
        cout<<"Time: "<<elapsedSeconds<<"s"<<endl;
        cout<<"Average fitness: "<<avgFitness<<endl<<endl;

        // Output an image with the best chromosome each 100 generations.
        if(generation % 100 == 0)
        {
            TImage result = getImageFromChromosome(&original, population[0].chromosome);
            write_jpeg_file("result.jpg", &result);

            delete [] result.data;
        }
        
        generation++;
	} 
	cout<<"Generation: "<<generation<<endl; 
	cout<<"Fitness: "<<population[0].fitness<<endl;

    TImage result = getImageFromChromosome(&original, population[0].chromosome);
    compressImage(&result, compressionRate);
    write_jpeg_file("result.jpg", &result);

    delete [] result.data;
    delete [] original.data;
} 

