const Discord = require('discord.js');
const client = new Discord.Client();
const auth = require('./auth.json');

client.on('ready', () => {console.log(`Logged in as ${client.user.tag}!`);});
client.login(auth.token);

// Reply to user
client.on('message', msg => {
    
    let id  = msg.member.user.tag
   
    const fs = require('fs') // Requiring fs module in which writeFile function is defined.

    // MODULAR PROGRAMMING METHODS

    function start_logging(){
        
        let logger = ""                                        // Setup Logger
        logger=msg.content
        console.log("==========================================")
        console.log(`START MODULE TRIGGERED . . . `);
        console.log("Sentence that has start word: "+logger+'.')
        console.log("==========================================")
        
        start_label = "[START]"

        fs.appendFile('Log.txt', start_label, (err) => {           // Write data in 'log.txt' . 
            if (err) throw err;                              // In case of a error throw err. 
        }) 

    }

    function stop_logging(){
        
        let logger = ""                                        // Setup Logger
        logger=msg.content
        console.log("==========================================")
        console.log(`STOP LOGGING . . . `);
        console.log("==========================================")
        
        start_label = "[STOP]"

        fs.appendFile('Log.txt', start_label, (err) => {           // Write data in 'log.txt' . 
            if (err) throw err;                              // In case of a error throw err. 
        }) 

    }


    function log(){
            //      LOG TERUS SAMPE ADA STOPWORD KETRIGGERED    //
    
            let dataset = ""                                        // Setup Logger
            dataset = msg.content+'.'
            console.log(msg.content)

            fs.appendFile('Log.txt', dataset, (err) => {           // Write data in 'log.txt' . 
                if (err) throw err;                                // In case of a error throw err. 
            }) 
    }


    //  DRIVER PROGRAM  
    
    // CAPTURING START WORDS IN THE CHATS
    
    var start_words = ["visit", "let's meet", "meet"]
    for (var i=0; i < start_words.length; i++) {
     if(msg.content.includes(start_words[i])) {
        start_logging()
     } //END-IF
    } //END-FOR

  
    
    //CAPTURING STOP WORDS IN THE CHATS
    var stop_words = ["OK", "ok","Ok"]
    for (var i=0; i < stop_words.length; i++) {
     if(msg.content.includes(stop_words[i])) {
        stop_logging()
     } //END-IF
    } //END-FOR


     // CAPTURING CHATS.
    if(!msg.content.includes(start_words)||!msg.content.includes(stop_words)){log()}


    // [START] chats. chats.  [STOP] --> harus filter lagi ini ntr


// UNUSED SECTION

// MISC - LOG ALL MESSAGES
    // if message not from bot 
    // if(msg.member.user.tag != "ScheduleQ#0119") {
    //    logger = "CHAT DATE : " + current_date + "/" + current_month + "/" + current_year + "\nCHAT TIME : "
    // + current_hour + ":" + current_minute + ":"+ current_sec + 
    // "\nMessage from user " + msg.member.user.tag + " : " +  msg.content + '\n\n' 
    //    console.log(`Chat from ${msg.member.user.tag} is stored at Log.txt`);
    // }
    // if message is from bot
    // else{
    //     logger = "CHAT DATE : " + current_date + "/" + current_month + "/" + current_year + "\nCHAT TIME : " + current_hour + ":" + current_minute + ":"+ current_sec + "\nMessage from bot " + '\n\n'
    //     console.log(`Chat from bot stored in Log.txt`);
    // }

    // Write data in 'log.txt' . 
    // fs.appendFile('Log.txt', logger, (err) => { 
        
    //     // In case of a error throw err. 
    //     if (err) throw err; 
    // }) 



  // MISC -  Setup Date & Time for Logging
    // var tanggal = new Date(); //date object
    // var current_date = tanggal.getDate();
    // var current_month = tanggal.getMonth()+1; //Month starts from 0
    // var current_year = tanggal.getFullYear();
    // var current_hour = tanggal.getHours()
    // var current_minute = tanggal.getMinutes()
    // var current_sec = tanggal.getSeconds();

    //END SECTION
});

