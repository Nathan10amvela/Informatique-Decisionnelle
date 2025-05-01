package com.example.ro.controllers;

import com.example.ro.dto.responseDTO.ApiError;
import com.example.ro.dto.requestDTO.PersonDTO;
import com.example.ro.dto.requestDTO.UpdatePersonDTO;
import com.example.ro.services.PersonService;
import jakarta.validation.Valid;
import lombok.AllArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/persons")
public class PersonController {

    private final PersonService personService;

    public PersonController(PersonService personService) {
        this.personService = personService;
    }




    @GetMapping("/{id}")
    public ApiError getPerson(@PathVariable int id) {
        return personService.getPersonById(id);
    }

    @PostMapping
    public ApiError createPerson(@Valid @RequestBody PersonDTO dto) {
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

    @GetMapping("/family-tree/{id}")
    public ApiError getPersonsByTreeId(@PathVariable int id) {return personService.getPersonsByTreeId(id); }
}

