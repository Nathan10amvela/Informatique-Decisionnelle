package com.example.ro.controllers;

import com.example.ro.dto.responseDTO.ApiError;
import com.example.ro.dto.requestDTO.PersonDTO;
import com.example.ro.dto.requestDTO.UpdatePersonDTO;
import com.example.ro.services.PersonService;
import lombok.AllArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/persons")
@AllArgsConstructor
public class PersonController {


    private final PersonService personService;


    @GetMapping("/{id}")
    public ApiError getPerson(@PathVariable int id) {
        return personService.getPersonById(id);
    }

    @PostMapping
    public ApiError createPerson(@RequestBody PersonDTO dto) {
        return personService.createPerson(dto);
    }

    @PatchMapping("/{id}")
    public ApiError updatePerson(@PathVariable int id, @RequestBody UpdatePersonDTO dto) {
        return personService.updatePerson(id, dto);
    }

    @DeleteMapping("/{id}")
    public ApiError deletePerson(@PathVariable int id) {
        return personService.deletePerson(id);
    }

    //TODO il faut créer la route qui doit permettre d'inviter un membre
}

