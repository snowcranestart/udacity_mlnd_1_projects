import webbrowser

class Movie():
    ''' Class Movie
    '''
    
    def __init__(self, movie_title, movie_storyline, poster_image, trailer_youtube):
        '''initiate a instance of class Movie, assign value for title,
           storyline, poster_iamge_url, trailer_youtube_url
           args:
           movie_title:the title of the movie
           movie_storyline:the storyline of the movie
           poster_iamge:the url of the poster iamge
           trailer_youtube:the url of the trailer_youtube
        '''
        
        self.title = movie_title
        self.storyline = movie_storyline
        self.poster_image_url = poster_image
        self.trailer_youtube_url = trailer_youtube

    def show_trailer(self):
        '''
        call the open function of webbrowser module to open the trailer of the movie
        '''

        webbrowser.open(self.trailer_youtube_url)
        
        
    
